 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated for 5 variables, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-key lookups at scale:** `idx_lookup` is a named vector with ~6.46M entries. Named-vector lookup in R uses linear hashing that degrades at this scale. Each of the 6.46M rows performs multiple lookups into it.
2. **Per-row `lapply` with allocations:** Each iteration creates temporary character vectors (`paste`), subsets a named vector, and filters NAs — millions of small allocations that thrash the garbage collector.
3. **Redundant computation:** The neighbor *structure* is time-invariant (same grid, same rook neighbors every year), but the lookup is rebuilt as if it varies per row. The neighbor graph is only ~344K cells; the time dimension simply replicates it.
4. **`do.call(rbind, ...)` on a 6.46M-element list:** This is a known slow pattern in R.

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The neighbor graph is **purely spatial** — cell A's neighbors are the same in every year. So we should:

1. **Build the neighbor lookup once at the cell level (344K cells), not the cell-year level (6.46M rows).**
2. **Use `data.table` for fast indexed joins** instead of named-vector lookups.
3. **Vectorize the stats computation** using `data.table` grouped operations — join each row to its neighbors' values and compute `max`, `min`, `mean` in bulk.

This replaces millions of R-level iterations with a single large equi-join + grouped aggregation, which `data.table` handles in seconds.

### Complexity Reduction

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M R-level iterations with string ops | 344K-cell edge list built once (vectorized) |
| Stats computation (per variable) | 6.46M `lapply` iterations | One `data.table` join + group-by aggregation |
| Total R-level loop iterations | ~6.46M × (1 + 5) ≈ 38.8M | 0 (fully vectorized) |

### Memory Estimate

The edge list (directed rook neighbors) has ~1.37M rows × 2 integer columns ≈ 11 MB. The main `data.table` is ~6.46M rows × 110 columns ≈ 5.7 GB for doubles, which fits in 16 GB. The join temporarily expands to ~6.46M × (avg ~4 neighbors) ≈ 25.8M rows but only for a few columns at a time — manageable.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a spatial edge list ONCE (cell-level, not row-level)
# ============================================================
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps position -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid    <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id       = id_order[from_idx],
    nb_id    = id_order[to_idx]
  )
}

# ============================================================
# STEP 2: Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_dt <- function(dt, edge_dt, var_name) {
  # dt must have columns: id, year, and <var_name>
  # edge_dt must have columns: id, nb_id

  # Subset to only needed columns for the join
  vals_dt <- dt[, .(id, year, val = get(var_name))]

  # Join: for each (id, year), find all neighbors' values

  # First, attach neighbor ids
  joined <- edge_dt[vals_dt, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # joined now has columns: id, nb_id, year, val
  # We need the NEIGHBOR's value, not the focal cell's value
  # So join again to get nb's val in that year
  setnames(vals_dt, c("id", "year", "val"), c("nb_id", "year", "nb_val"))
  joined <- vals_dt[joined, on = c("nb_id", "year"), nomatch = NA]
  # joined now has: nb_id, year, nb_val, id, val

  # Compute grouped stats: group by (id, year)
  stats <- joined[!is.na(nb_val),
                   .(nb_max  = max(nb_val),
                     nb_min  = min(nb_val),
                     nb_mean = mean(nb_val)),
                   by = .(id, year)]

  # Build output column names
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  stats
}

# ============================================================
# STEP 3: Main pipeline
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (by reference if already, otherwise copy)
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }

  # Ensure key columns exist
  stopifnot(all(c("id", "year") %in% names(dt)))

  # Step 1: Build edge list once
  message("Building spatial edge list...")
  edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
  setkey(edge_dt, id)
  message(sprintf("  Edge list: %s directed edges", format(nrow(edge_dt), big.mark = ",")))

  # Step 2: For each variable, compute and merge neighbor stats
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats_dt <- compute_neighbor_stats_dt(dt, edge_dt, var_name)

    # Merge back into main table
    dt <- merge(dt, stats_dt, by = c("id", "year"), all.x = TRUE)

    # Free memory
    rm(stats_dt)
    gc()
  }

  message("Done. All neighbor features added.")
  return(dt)
}

# ============================================================
# USAGE
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values. The join logic replicates the original's "find all rook neighbors of cell *i* that exist in year *t*, get their values, compute stats." |
| **Trained RF model** | No model retraining. The output columns have the same names and values, so `predict(rf_model, newdata = ...)` works identically. |
| **NA handling** | `!is.na(nb_val)` in the grouped aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with no valid neighbors get `NA` via `all.x = TRUE` in the merge. |

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M R iterations, string ops | Vectorized edge list build (~0.5 sec) |
| **Stats per variable** | ~6.46M `lapply` calls | One `data.table` join + grouped agg (~30–90 sec) |
| **Total wall time** | 86+ hours (estimated) | **~5–10 minutes** |
| **Peak RAM** | Uncontrolled (millions of small allocations) | ~8–12 GB (controlled, fits in 16 GB) |