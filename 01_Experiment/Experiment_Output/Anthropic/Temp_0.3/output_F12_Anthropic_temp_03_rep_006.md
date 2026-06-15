 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable — only called 5 times total. This is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Row-level `paste()` and character lookups inside `lapply` over 6.46 million rows.** Each iteration of the `lapply` in `build_neighbor_lookup` performs:
   - `as.character(data$id[i])` — character conversion per row.
   - `id_to_ref[as.character(...)]` — named-vector lookup (hash-like, but with overhead per call).
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` — string construction for every neighbor of every row.
   - `idx_lookup[neighbor_keys]` — another named-vector lookup on a 6.46M-entry vector, repeated for each neighbor key.

2. **Scale of the problem:** With ~6.46 million rows and an average of ~4 rook neighbors per cell, this inner function performs roughly **25.8 million `paste()` calls and 25.8 million named-vector lookups**, all inside a sequential `lapply`. String construction and hashing at this scale dominate runtime.

3. **`compute_neighbor_stats()`**, by contrast, does only integer indexing (`vals[idx]`) and simple arithmetic — these are fast vectorized operations. The `do.call(rbind, result)` on a list of 6.46M length-3 vectors takes seconds, not hours.

**Conclusion:** The 86+ hour runtime is driven by the massive string-based lookup construction in `build_neighbor_lookup()`, not by `do.call(rbind, ...)` or list binding in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all string operations from the lookup.** Replace `paste(id, year)` key construction with direct integer arithmetic. Map `(id, year)` pairs to row indices using an integer-keyed structure (a matrix or `data.table` join) instead of a named character vector.

2. **Vectorize the neighbor lookup construction.** Instead of `lapply` over 6.46M rows, expand the neighbor relationships into a flat table, join on `(neighbor_id, year)` to get target row indices, then split back into a list. This replaces millions of R-level function calls with a single `data.table` merge.

3. **Vectorize `compute_neighbor_stats()`.** Instead of `lapply` + `do.call(rbind, ...)`, use `data.table` grouped aggregation on the flat edge list to compute max/min/mean in one pass per variable.

4. **Preserve the trained Random Forest model and original numerical estimand.** The output columns are identical — same neighbor max, min, mean values — just computed faster.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a fast integer-indexed row lookup using data.table
# ---------------------------------------------------------------
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs (index corresponds to nb object position)
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Create a mapping from (id, year) -> row_idx
  setkey(dt, id, year)

  # --- Build flat edge list: for each cell position in id_order,
  #     enumerate its neighbor cell IDs ---
  # Convert nb object to a flat edge list (focal_pos -> neighbor_pos)
  n_cells <- length(id_order)
  focal_pos <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_pos <- unlist(neighbors)

  # Remove zero-length entries (isolated cells produce integer(0))
  valid <- !is.na(neighbor_pos) & neighbor_pos > 0
  focal_pos <- focal_pos[valid]
  neighbor_pos <- neighbor_pos[valid]

  # Map positions to actual cell IDs
  focal_ids <- id_order[focal_pos]
  neighbor_ids <- id_order[neighbor_pos]

  edge_dt <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)

  # --- Cross with years to get (focal_id, year, neighbor_id, year) ---
  years <- sort(unique(dt$year))

  # Expand edges across all years
  # This creates ~1.37M edges * 28 years ≈ 38.5M rows — fits in 16GB RAM
  edge_year <- CJ_dt(edge_dt, years)

  # Join to get neighbor row indices
  setkey(dt, id, year)
  edge_year[dt, neighbor_row_idx := i.row_idx,
            on = .(neighbor_id = id, year = year)]

  # Join to get focal row indices
  edge_year[dt, focal_row_idx := i.row_idx,
            on = .(focal_id = id, year = year)]

  # Drop edges where either side has no matching row
  edge_year <- edge_year[!is.na(focal_row_idx) & !is.na(neighbor_row_idx)]

  return(edge_year)
}

# Helper: cross join a data.table with a vector of years
CJ_dt <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join
  k <- nrow(edge_dt)
  m <- length(years)
  out <- data.table(
    focal_id    = rep(edge_dt$focal_id, each = m),
    neighbor_id = rep(edge_dt$neighbor_id, each = m),
    year        = rep(years, times = k)
  )
  return(out)
}

# ---------------------------------------------------------------
# STEP 2: Vectorized neighbor stats using data.table grouping
# ---------------------------------------------------------------
compute_neighbor_stats_fast <- function(cell_data, edge_year, var_name) {
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Attach the variable values for neighbor rows
  vals <- dt[[var_name]]
  edge_year[, neighbor_val := vals[neighbor_row_idx]]

  # Group by focal_row_idx and compute stats
  stats <- edge_year[!is.na(neighbor_val),
                     .(nmax  = max(neighbor_val),
                       nmin  = min(neighbor_val),
                       nmean = mean(neighbor_val)),
                     by = focal_row_idx]

  # Initialize output columns with NA
  n <- nrow(dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  col_max[stats$focal_row_idx]  <- stats$nmax
  col_min[stats$focal_row_idx]  <- stats$nmin
  col_mean[stats$focal_row_idx] <- stats$nmean

  # Clean up temp column
  edge_year[, neighbor_val := NULL]

  suffix <- paste0("n_", var_name)
  cell_data[[paste0(suffix, "_max")]]  <- col_max
  cell_data[[paste0(suffix, "_min")]]  <- col_min
  cell_data[[paste0(suffix, "_mean")]] <- col_mean

  return(cell_data)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline (replaces the original outer loop)
# ---------------------------------------------------------------
# Build the flat edge-year table ONCE (replaces build_neighbor_lookup)
message("Building vectorized neighbor edge table...")
edge_year <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_neighbor_stats_fast(cell_data, edge_year, var_name)
}

message("Done. Proceed to prediction with the existing trained Random Forest model.")
# The trained RF model is untouched — use predict(rf_model, cell_data) as before.
```

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| **Lookup construction** | ~6.46M `lapply` iterations with `paste()` + named-vector hash lookups → **dominant bottleneck (hours)** | Single `data.table` cross-join + keyed merge → **minutes** |
| **Neighbor stats** | `lapply` + `do.call(rbind, ...)` × 5 vars → moderate cost | `data.table` grouped aggregation × 5 vars → **seconds** |
| **Memory** | ~6.46M-element list of integer vectors | ~38.5M-row flat `data.table` (~1–2 GB, fits in 16 GB) |
| **Estimated total time** | 86+ hours | **~5–15 minutes** |

The numerical output (neighbor max, min, mean per variable per cell-year) is identical, and the trained Random Forest model is completely preserved.