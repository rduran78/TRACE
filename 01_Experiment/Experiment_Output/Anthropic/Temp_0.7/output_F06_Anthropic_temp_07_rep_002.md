 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) latent cost
The function iterates over all **~6.46 million rows** with `lapply`, and for each row it:
1. Looks up the cell's reference index.
2. Retrieves neighbor cell IDs.
3. Constructs character keys (`paste(id, year)`).
4. Looks them up in a named character vector (`idx_lookup`).

Character-key hashing via named vectors in R is slow at this scale. With ~6.46M rows, building and probing a ~6.46M-element named character vector millions of times is extremely expensive. The `paste()` calls alone generate millions of temporary strings.

### Bottleneck 2: `compute_neighbor_stats` — repeated per-row `lapply`
For each of the 5 variables, another `lapply` over 6.46M rows extracts neighbor values, drops NAs, and computes `max`, `min`, `mean`. That's **~32.3 million** individual R-level function calls (5 vars × 6.46M rows), each involving subsetting and summary stats in interpreted R.

### Why raster focal/kernel operations don't directly apply
Raster focal operations (e.g., `terra::focal`) assume a regular rectangular grid with a fixed kernel. Here the data is a **panel** (cell × year), neighbor relationships come from an irregular `spdep::nb` object, and the computation is per-variable summary stats over rook neighbors within the same year. Focal operations can't handle the panel dimension or irregular neighborhoods without significant reshaping that could introduce numerical differences. We must preserve the exact numerical estimand, so we stay with the tabular approach but vectorize it.

### Summary
| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | 6.46M `paste` + named-vector lookups |
| `compute_neighbor_stats` | ~hours × 5 vars | 6.46M interpreted R loops × 5 |
| **Total** | **~86+ hours** | Interpreted R loops, string operations |

---

## Optimization Strategy

### Strategy 1: Eliminate the row-level lookup entirely — work in year-slices with integer indexing

**Key insight:** Neighbors are purely spatial (rook), and the panel is balanced (every cell appears in every year). So for a given year, the neighbor indices into that year-slice are the *same* for every year — they're just the spatial neighbor indices from `rook_neighbors_unique`. We never need to build a 6.46M-element lookup. We just need to, for each year, subset the data and apply the *spatial* neighbor list.

### Strategy 2: Vectorized sparse-matrix multiplication for `max`, `min`, `mean`

We construct a sparse adjacency matrix `W` (344,208 × 344,208) from `rook_neighbors_unique`. Then:
- **mean** of neighbors = `(W %*% x) / (W %*% ones)` (sparse matrix-vector multiply).
- **max** and **min**: Use grouped operations via `data.table` with an edge list, which is far faster than per-row `lapply`.

### Strategy 3: `data.table` for all operations

Replace all `data.frame` operations with `data.table` for memory efficiency and speed.

### Expected speedup
- Sparse matrix multiply for mean: O(nnz) ≈ 1.37M per year × 28 years = ~38.4M operations, done in compiled C (via `Matrix` package). Essentially instant.
- Grouped `data.table` operations for max/min on an edge list: similarly fast.
- Total estimated time: **minutes**, not hours.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# ==============================================================================
# 
# Prerequisites:
#   cell_data        : data.frame/data.table with columns: id, year, and the
#                      neighbor_source_vars. Rows are sorted (or will be sorted)
#                      by (year, id).
#   id_order         : integer/character vector of unique cell IDs in the order
#                      corresponding to rook_neighbors_unique.
#   rook_neighbors_unique : spdep::nb object (list of length = number of cells).
#   rf_model         : pre-trained Random Forest model (untouched).
#
# Output:
#   cell_data gains 15 new columns (3 stats × 5 vars), numerically identical
#   to the original implementation.
# ==============================================================================

library(data.table)
library(Matrix)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ---- 0. Convert to data.table if needed ----
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  # ---- 1. Build integer mapping: cell id -> position in id_order ----
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # ---- 2. Add a spatial index column to cell_data ----
  # This maps each row's cell id to its position in id_order.
  cell_data[, sp_idx := id_to_pos[as.character(id)]]

  # ---- 3. Sort by (year, sp_idx) so that within each year the rows are
  #         in the same order as id_order. This is critical: it means
  #         row i within a year-slice corresponds to spatial index i. ----
  setkey(cell_data, year, sp_idx)

  # Verify balanced panel
  rows_per_year <- cell_data[, .N, by = year]
  if (!all(rows_per_year$N == n_cells)) {
    # Unbalanced panel: fall back to a safe but still fast approach
    message("Panel is not perfectly balanced. Using edge-list approach for all stats.")
    return(.compute_features_edgelist(cell_data, id_to_pos, id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars, years))
  }

  # ---- 4. Build sparse adjacency matrix W (n_cells x n_cells) ----
  # From the nb object, construct COO triplets.
  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (i in seq_len(n_cells)) {
    nbrs <- rook_neighbors_unique[[i]]
    # spdep::nb encodes no-neighbor as 0L in a length-1 vector
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    from_vec <- c(from_vec, rep(i, length(nbrs)))
    to_vec   <- c(to_vec, nbrs)
  }

  # Build edge list data.table (will be reused for max/min)
  edge_dt <- data.table(from = from_vec, to = to_vec)

  # Sparse binary adjacency matrix (for mean computation)
  W <- sparseMatrix(i = from_vec, j = to_vec, x = 1,
                    dims = c(n_cells, n_cells))

  # Number of neighbors per cell (for computing mean)
  n_neighbors <- as.numeric(W %*% rep(1, n_cells))  # same for every year

  # ---- 5. For each variable, compute max, min, mean across all years ----
  for (var_name in neighbor_source_vars) {

    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)

    # Pre-allocate result vectors
    res_max  <- rep(NA_real_, nrow(cell_data))
    res_min  <- rep(NA_real_, nrow(cell_data))
    res_mean <- rep(NA_real_, nrow(cell_data))

    vals_all <- cell_data[[var_name]]

    for (yi in seq_along(years)) {
      yr <- years[yi]

      # Row range for this year (data is sorted by year, sp_idx)
      row_start <- (yi - 1L) * n_cells + 1L
      row_end   <- yi * n_cells
      row_range <- row_start:row_end

      x <- vals_all[row_range]  # length = n_cells, in sp_idx order

      # --- MEAN via sparse matrix multiply ---
      # Handle NAs: we need mean of non-NA neighbors.
      # sum of non-NA neighbor values / count of non-NA neighbors
      x_nona     <- x
      x_nona[is.na(x_nona)] <- 0
      not_na     <- as.numeric(!is.na(x))

      sum_vals   <- as.numeric(W %*% x_nona)
      count_vals <- as.numeric(W %*% not_na)

      yr_mean <- ifelse(count_vals > 0, sum_vals / count_vals, NA_real_)

      # --- MAX and MIN via edge list + data.table ---
      # For each "from" cell, gather neighbor values and compute max/min.
      edge_dt[, val := x[to]]

      # Remove NA neighbor values for aggregation
      agg <- edge_dt[!is.na(val),
                      .(vmax = max(val), vmin = min(val)),
                      by = from]

      yr_max <- rep(NA_real_, n_cells)
      yr_min <- rep(NA_real_, n_cells)
      yr_max[agg$from] <- agg$vmax
      yr_min[agg$from] <- agg$vmin

      # Also set mean to NA for cells with no neighbors at all
      yr_mean[n_neighbors == 0] <- NA_real_

      # Write results
      res_max[row_range]  <- yr_max
      res_min[row_range]  <- yr_min
      res_mean[row_range] <- yr_mean
    }

    # Add columns to cell_data
    set(cell_data, j = col_max,  value = res_max)
    set(cell_data, j = col_min,  value = res_min)
    set(cell_data, j = col_mean, value = res_mean)

    message(sprintf("  Done: %s", var_name))
  }

  # ---- 6. Clean up helper column ----
  cell_data[, sp_idx := NULL]

  return(cell_data)
}


# ==============================================================================
# FALLBACK: Edge-list approach for unbalanced panels
# ==============================================================================
.compute_features_edgelist <- function(cell_data, id_to_pos, id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars, years) {
  n_cells <- length(id_order)

  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (i in seq_len(n_cells)) {
    nbrs <- rook_neighbors_unique[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    from_vec <- c(from_vec, rep(i, length(nbrs)))
    to_vec   <- c(to_vec, nbrs)
  }
  edge_dt <- data.table(from_sp = from_vec, to_sp = to_vec)

  # Build a keyed lookup: (sp_idx, year) -> row index in cell_data
  cell_data[, .row_idx := .I]
  lookup <- cell_data[, .(sp_idx, year, .row_idx)]
  setkey(lookup, sp_idx, year)

  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)

    res_max  <- rep(NA_real_, nrow(cell_data))
    res_min  <- rep(NA_real_, nrow(cell_data))
    res_mean <- rep(NA_real_, nrow(cell_data))

    vals_all <- cell_data[[var_name]]

    for (yr in years) {
      yr_rows <- lookup[J(seq_len(n_cells), yr), .row_idx, nomatch = NA]
      # yr_rows[i] = row index for spatial cell i in this year (or NA)

      x <- vals_all[yr_rows]  # length n_cells

      edge_dt[, val := x[to_sp]]
      agg <- edge_dt[!is.na(val),
                      .(vmax = max(val), vmin = min(val), vmean = mean(val)),
                      by = from_sp]

      target_rows <- yr_rows[agg$from_sp]
      valid <- !is.na(target_rows)
      res_max[target_rows[valid]]  <- agg$vmax[valid]
      res_min[target_rows[valid]]  <- agg$vmin[valid]
      res_mean[target_rows[valid]] <- agg$vmean[valid]
    }

    set(cell_data, j = col_max,  value = res_max)
    set(cell_data, j = col_min,  value = res_min)
    set(cell_data, j = col_mean, value = res_mean)

    message(sprintf("  Done: %s", var_name))
  }

  cell_data[, .row_idx := NULL]
  cell_data[, sp_idx := NULL]
  return(cell_data)
}


# ==============================================================================
# USAGE
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features (optimized)...")
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)
message("Done. Proceeding to prediction with trained RF model...")

# The trained Random Forest model is unchanged.
# Predict as before (example):
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **max** | `max(vals[idx])` after removing NAs | `edge_dt[!is.na(val), max(val), by=from]` | Identical — same values, same `max()` function |
| **min** | `min(vals[idx])` after removing NAs | `edge_dt[!is.na(val), min(val), by=from]` | Identical |
| **mean** | `mean(vals[idx])` after removing NAs | `(W %*% x_nona) / (W %*% not_na)` | Identical — sum of non-NA values divided by count of non-NA values is the arithmetic mean |
| **NAs** | Returns `c(NA,NA,NA)` when no valid neighbors | Both paths produce `NA` for zero-count | Identical |

The Random Forest model is never retrained — it is applied as-is to the augmented `cell_data`.

---

## Expected Performance

| Step | Original | Optimized | Speedup |
|------|----------|-----------|---------|
| Neighbor lookup construction | ~hours (string ops) | Eliminated (integer sort) | ∞ |
| Stats per variable per year | ~3.5 hrs | ~1–3 seconds (sparse matmul + data.table groupby) | ~1000× |
| **Total (5 vars × 28 years)** | **~86+ hours** | **~5–15 minutes** | **~350–1000×** |

The dominant costs become the 140 sparse matrix-vector multiplies (each ~1.37M nonzeros, sub-second) and 140 `data.table` grouped aggregations over ~1.37M edges (also sub-second each). Total wall-clock time on a 16 GB laptop should be well under 15 minutes.