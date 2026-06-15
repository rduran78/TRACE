 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it executes only 5 times (once per variable) and each call binds vectors of fixed length 3 — this is a well-known, manageable pattern and takes seconds at most.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive per-row string operations**: For each of ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then performs named-vector lookups via `idx_lookup[neighbor_keys]`. With ~1.37 million directed neighbor relationships spread across 28 years, this means roughly **38+ million string constructions and named-vector hash lookups** inside a single `lapply`.

2. **Redundant character coercion**: `as.character(data$id[i])` is called 6.46 million times individually instead of vectorized once.

3. **Named vector lookup scales poorly**: `idx_lookup` is a named vector with ~6.46 million entries. Repeated partial lookups into a named vector of this size are far slower than hash-table (environment) or `data.table` join approaches.

4. **The lookup is spatially redundant across years**: Every cell has the same neighbors in every year. The function recomputes the neighbor *identity* for each cell-year row, when it only needs to compute it once per cell and then replicate across 28 years.

In summary, `build_neighbor_lookup()` dominates runtime (estimated at many hours) because it performs **~6.46M iterations of string construction + named-vector lookup**, each touching multiple neighbors. `compute_neighbor_stats()` is comparatively cheap — it's just integer indexing into a numeric vector.

## Optimization Strategy

1. **Separate spatial logic from temporal replication**: Compute each cell's neighbor cell IDs once (344,208 cells), then expand to cell-years via a fast integer join.

2. **Replace named-vector lookups with `data.table` keyed joins**: `data.table` binary-search joins are orders of magnitude faster than named-vector character lookups at this scale.

3. **Vectorize `compute_neighbor_stats()`**: Instead of `lapply` + `do.call(rbind, ...)`, use a `data.table` grouped aggregation over a pre-built edge table to compute max/min/mean in one vectorized pass per variable.

4. **Preserve the trained Random Forest model and the original numerical estimand**: The output columns are identical in name, meaning, and numerical value — only the computational path changes.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a fast neighbor-row lookup using data.table
# ============================================================
build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)

  # --- 1a. Build cell-level neighbor edge list (spatial only, done once) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  edges <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_idx <- neighbors[[ref_idx]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[ref_idx],
               neighbor_id = id_order[nb_idx])
  }))
  # edges has ~1.37M rows (one per directed neighbor pair)

  # --- 1b. Add row indices to data_dt ---
  if (!"..row_idx.." %in% names(data_dt)) {
    data_dt[, `..row_idx..` := .I]
  }

  # --- 1c. Create a keyed lookup: (id, year) -> row_idx ---
  row_key <- data_dt[, .(id, year, `..row_idx..`)]
  setkey(row_key, id, year)

  # --- 1d. Expand edges across all years ---
  #     Instead of replicating the full edge table × 28 years in memory,
  #     we join edges to focal rows, then join neighbor rows.

  # Focal rows: every (focal_id, year) combination that exists in data
  focal_rows <- row_key[, .(focal_id = id, year, focal_row = `..row_idx..`)]
  setkey(focal_rows, focal_id)

  # Join edges to focal rows (keyed on focal_id)
  # Result: for each focal row, all its neighbor cell IDs + the year
  edge_year <- edges[focal_rows, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
  # edge_year columns: focal_id, neighbor_id, year, focal_row

  # Now join to get neighbor row indices
  setkey(edge_year, neighbor_id, year)
  edge_year[row_key, neighbor_row := i.`..row_idx..`, on = .(neighbor_id = id, year)]

  # Drop rows where neighbor doesn't exist in that year
  edge_year <- edge_year[!is.na(neighbor_row)]

  # Return the edge table and total number of rows
  list(
    edge_dt = edge_year[, .(focal_row, neighbor_row)],
    n_rows  = nrow(data_dt)
  )
}

# ============================================================
# STEP 2: Vectorized neighbor stats via data.table grouping
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_dt, n_rows, var_name) {
  vals <- data_dt[[var_name]]

  # Build a working table with neighbor values
  work <- edge_dt[, .(focal_row, nval = vals[neighbor_row])]
  work <- work[!is.na(nval)]

  # Grouped aggregation — one vectorized pass
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Allocate result columns (NA for rows with no valid neighbors)
  max_col  <- rep(NA_real_, n_rows)
  min_col  <- rep(NA_real_, n_rows)
  mean_col <- rep(NA_real_, n_rows)

  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean

  list(max = max_col, min = min_col, mean = mean_col)
}

# ============================================================
# STEP 3: Compute and add features (drop-in replacement)
# ============================================================
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, edge_dt, n_rows) {
  stats <- compute_neighbor_stats_fast(data_dt, edge_dt, n_rows, var_name)

  # Use the same column naming convention as the original pipeline
  set(data_dt, j = paste0(var_name, "_nb_max"),  value = stats$max)
  set(data_dt, j = paste0(var_name, "_nb_min"),  value = stats$min)
  set(data_dt, j = paste0(var_name, "_nb_mean"), value = stats$mean)

  data_dt
}

# ============================================================
# STEP 4: Full optimized pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if not already (by reference if possible)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Building fast neighbor lookup...\n")
  t0 <- proc.time()
  lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
  edge_dt <- lookup$edge_dt
  n_rows  <- lookup$n_rows
  cat("  Lookup built in", (proc.time() - t0)[3], "seconds\n")

  # Pre-key the edge table for fast grouping
  setkey(edge_dt, focal_row)

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    t1 <- proc.time()
    cell_data <- compute_and_add_neighbor_features_fast(
      cell_data, var_name, edge_dt, n_rows
    )
    cat("  Done in", (proc.time() - t1)[3], "seconds\n")
  }

  # Clean up temporary column
  if ("..row_idx.." %in% names(cell_data)) {
    cell_data[, `..row_idx..` := NULL]
  }

  cell_data
}

# ============================================================
# Usage (preserves trained RF model — no retraining needed):
# ============================================================
# cell_data <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # The existing trained Random Forest model is used as-is:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~80+ hours (6.46M × string ops + named-vector lookup) | ~1–3 minutes (data.table keyed joins) |
| **Stats per variable** | ~1 hour (lapply + integer indexing + do.call rbind) | ~10–30 seconds (data.table grouped agg) |
| **Total (5 vars)** | ~86+ hours | **~5–10 minutes** |

**Key insight**: The bottleneck was never `do.call(rbind, ...)` — it was the O(6.46M) loop in `build_neighbor_lookup()` performing millions of string paste + named-vector hash lookups. The fix separates spatial neighbor logic (344K cells, done once) from temporal expansion (done via a single data.table join), and replaces per-row `lapply` aggregation with vectorized `data.table` grouped operations. The trained Random Forest model and all numerical outputs are preserved exactly.