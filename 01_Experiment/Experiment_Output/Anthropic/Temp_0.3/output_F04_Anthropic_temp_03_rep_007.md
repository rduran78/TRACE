 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks up those keys in `idx_lookup` (a named character vector).

With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates and searches **~25.8 million paste/key-lookup operations** inside an R-level loop — all interpreted, no vectorization. `compute_neighbor_stats` then loops again over 6.46M entries, but is lighter. The combined cost explains the 86+ hour estimate.

**Root causes:**
- Row-level `lapply` in pure R over millions of rows.
- Repeated `paste()` string construction and named-vector lookup (O(n) or hash overhead per call).
- `compute_neighbor_stats` uses another `lapply` + `do.call(rbind, ...)` over 6.46M small vectors.

## Optimization Strategy

**Replace the row-level loop with a vectorized, year-grouped merge + `data.table` aggregation.**

Key insight: For a given year, every cell's neighbors are the same set of cell IDs (from the static `rook_neighbors_unique` nb object). So we can:

1. Build an **edge list** from the nb object once (source_id → neighbor_id), ~1.37M edges.
2. For each year, join the edge list to the data to retrieve neighbor variable values — this is a vectorized merge.
3. Aggregate (max, min, mean) by (source_id, year) using `data.table` grouped operations.

This eliminates all per-row R-level loops and string key construction. Expected speedup: **~100–500x** (minutes instead of days).

## Optimized R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a static edge list from the nb object (run once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  n <- length(neighbors)
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  src <- integer(n_edges)
  dst <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    len <- length(nb_i)
    if (len > 0L) {
      src[pos:(pos + len - 1L)] <- id_order[i]
      dst[pos:(pos + len - 1L)] <- id_order[nb_i]
      pos <- pos + len
    }
  }
  data.table(source_id = src, neighbor_id = dst)
}

# ---------------------------------------------------------------
# 2. Compute neighbor features for one variable (vectorized)
# ---------------------------------------------------------------
compute_neighbor_features_fast <- function(dt, edge_dt, var_name) {
  # dt is a data.table with columns: id, year, <var_name>
  # edge_dt is data.table(source_id, neighbor_id)

  # Subset to needed columns for the join
  vals_dt <- dt[, .(neighbor_id = id, year, val = get(var_name))]

  # Join: for each (source_id, year), look up each neighbor's value
  # Keyed join on (neighbor_id, year)
  setkey(vals_dt, neighbor_id, year)
  # Expand edges by year via join
  joined <- edge_dt[vals_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # joined now has columns: source_id, neighbor_id, year, val

  # Remove NAs in val before aggregation
  joined <- joined[!is.na(val)]

  # Aggregate by (source_id, year)
  agg <- joined[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(source_id, year)]

  # Rename to match variable-specific column names
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  agg
}

# ---------------------------------------------------------------
# 3. Main pipeline (replaces the original outer loop)
# ---------------------------------------------------------------
# Convert to data.table if not already
cell_dt <- as.data.table(cell_data)

# Build edge list once (~1.37M rows)
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge neighbor features for each variable
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  agg <- compute_neighbor_features_fast(cell_dt, edge_dt, var_name)
  # Merge back onto main table
  cell_dt <- merge(cell_dt, agg, by.x = c("id", "year"), by.y = c("source_id", "year"), all.x = TRUE)
  rm(agg)
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The aggregation computes the identical `max`, `min`, `mean` of non-NA neighbor values per (cell, year). No approximation. |
| **Trained RF model** | No model retraining; we only reconstruct the same input features the model expects. Column names follow the same `nb_max_<var>`, `nb_min_<var>`, `nb_mean_<var>` convention (adjust `setnames` if the original used different names). |
| **Missing data handling** | `!is.na(val)` filtering and `all.x = TRUE` merge produce `NA` for cells with zero valid neighbors, matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup / edge list | ~hours (6.46M `paste` + lookup) | **~2 seconds** (vectorized pre-allocation) |
| Neighbor stats (×5 vars) | ~days (6.46M × 5 `lapply`) | **~2–5 minutes** (`data.table` keyed join + grouped agg) |
| **Total** | **~86+ hours** | **~5–10 minutes** |

The `data.table` keyed join on `(neighbor_id, year)` is O(n log n) and cache-friendly, and the grouped aggregation is internally parallelized in C. Peak memory for the largest intermediate (`joined`) is roughly 1.37M edges × 28 years × 3 columns ≈ ~300 MB, well within 16 GB RAM.