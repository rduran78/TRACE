 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable — only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning pre-allocated length-3 vectors. That part is O(n) and relatively efficient.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive character key construction and named-vector lookup**: For each of the ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build character keys, then does named-vector indexing into `idx_lookup` (itself a named vector of length ~6.46 million). Named-vector lookup in R is **O(n)** per query (linear scan or hash with overhead), and this is done for every neighbor of every row. With ~1.37 million directed neighbor relationships spread across 28 years, this produces roughly **38.4 million key lookups** into a 6.46-million-entry named vector — all inside a sequential `lapply` with per-element `paste` and `as.character` overhead.

2. **Redundant recomputation across years**: The spatial neighbor structure is **time-invariant** — cell A's rook neighbors are the same in every year. Yet `build_neighbor_lookup` recomputes the neighbor-row mapping for every cell-year row independently, doing 28× the necessary spatial work.

3. **Per-element R-level overhead**: The `lapply` over 6.46 million rows with multiple character operations, subsetting, and `is.na` filtering per iteration incurs enormous interpreter overhead.

The `compute_neighbor_stats` function, by contrast, does only cheap integer-vector subsetting (`vals[idx]`) and simple numeric operations — it is fast once the lookup exists.

## Optimization Strategy

1. **Build the lookup using integer indexing via merge/join, not character key hashing.** Use `data.table` to create a fast equi-join between (neighbor_id, year) and (id, year) to resolve row indices in vectorized, C-level code.

2. **Exploit time-invariance of the spatial structure.** Build the spatial neighbor pairs once (344,208 cells × their neighbors ≈ 1.37M pairs), then cross-join with years in a single vectorized operation.

3. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation instead of row-wise `lapply`.

4. **Preserve the trained Random Forest model** — we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build neighbor lookup via vectorized data.table join
#    (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index
  # Ensure row_idx exists
  data_dt[, row_idx := .I]

  # --- Step A: Build spatial neighbor edge list (time-invariant) ---
  # neighbors is an nb object: list of integer index vectors into id_order
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nbrs <- neighbors[[i]]
    if (length(nbrs) == 0L || (length(nbrs) == 1L && nbrs[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nbrs])
  }))

  # --- Step B: Cross with years via join ---
  # Create keyed version of data for joining
  data_key <- data_dt[, .(id, year, row_idx)]
  setkey(data_key, id, year)

  # Get focal row indices
  focal_key <- copy(data_key)
  setnames(focal_key, c("id", "year", "row_idx"), c("focal_id", "year", "focal_row"))

  # Get neighbor row indices
  nbr_key <- copy(data_key)
  setnames(nbr_key, c("id", "year", "row_idx"), c("neighbor_id", "year", "nbr_row"))

  # Join: for each edge × year, get focal_row and nbr_row
  setkey(edge_list, focal_id)
  setkey(focal_key, focal_id, year)

  # Expand edges across all years of the focal cell
  edges_with_focal <- merge(edge_list, focal_key, by = "focal_id", allow.cartesian = TRUE)

  # Now resolve neighbor rows
  setkey(edges_with_focal, neighbor_id, year)
  setkey(nbr_key, neighbor_id, year)

  edges_full <- merge(edges_with_focal, nbr_key, by = c("neighbor_id", "year"), nomatch = 0L)

  # Return the full edge table: focal_row -> nbr_row
  edges_full[, .(focal_row, nbr_row)]
}

# ---------------------------------------------------------------
# 2. Vectorized neighbor stats via data.table grouping
#    (replaces compute_neighbor_stats + do.call(rbind,...))
# ---------------------------------------------------------------
compute_neighbor_stats_fast <- function(data_dt, edges_dt, var_name) {
  # edges_dt has columns: focal_row, nbr_row
  # Pull neighbor values via integer indexing (vectorized)
  vals <- data_dt[[var_name]]
  work <- edges_dt[, nbr_val := vals[nbr_row]]

  # Remove NAs in neighbor values
  work <- work[!is.na(nbr_val)]

  # Grouped aggregation — all in C-level data.table code
  stats <- work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = focal_row]

  # Build output columns aligned to all rows
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row]  <- stats$nb_max
  out_min[stats$focal_row]  <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  list(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# ---------------------------------------------------------------
# 3. Main pipeline (replaces outer loop)
# ---------------------------------------------------------------
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if needed (non-destructive copy)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  message("Building vectorized neighbor edge table...")
  t0 <- Sys.time()
  edges_dt <- build_neighbor_lookup_fast(dt, id_order, rook_neighbors_unique)
  message("  Edge table built: ", nrow(edges_dt), " directed cell-year edges in ",
          round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    stats <- compute_neighbor_stats_fast(dt, copy(edges_dt), var_name)

    # Assign columns with same naming convention as original code
    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(dt, j = col_max,  value = stats$nb_max)
    set(dt, j = col_min,  value = stats$nb_min)
    set(dt, j = col_mean, value = stats$nb_mean)

    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
  }

  # Remove helper column and return as data.frame to preserve downstream compatibility
  dt[, row_idx := NULL]
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# Usage (drop-in replacement):
# ---------------------------------------------------------------
# cell_data <- run_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then predict with the EXISTING trained model — no retraining:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M `lapply` iterations with `paste`/named-vector lookup | Eliminated |
| **Key construction** | Character `paste` + named-vector search (R-level, per row) | Integer equi-join via `data.table` merge (C-level, vectorized) |
| **Year redundancy** | Recomputes spatial structure 28× | Builds spatial edges once, cross-joins with years |
| **Stats computation** | `lapply` over 6.46M rows + `do.call(rbind,...)` | `data.table` grouped `max/min/mean` (single pass, C-level) |
| **Estimated runtime** | 86+ hours | ~5–15 minutes on the same laptop |
| **Numerical output** | Identical `max`, `min`, `mean` per cell-year | Identical (preserved) |
| **Trained RF model** | Untouched | Untouched |

The colleague was looking at the wrong function. The real cost was the **6.46-million-iteration, character-key-based, year-redundant spatial lookup construction** in `build_neighbor_lookup()`, not the relatively lightweight `do.call(rbind, ...)` assembly in `compute_neighbor_stats()`.