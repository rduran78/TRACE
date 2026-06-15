 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### The Core Problem: Row-by-Row `paste`+`match` Inside `lapply` Over 6.46M Rows

1. **`build_neighbor_lookup`** iterates over every row (`~6.46M`) and, for each row, constructs string keys via `paste()` and looks them up in a named vector (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-based but still carries per-call overhead. With ~6.46M iterations × ~4 neighbors on average, that's ~25.8M `paste` + hash-probe operations. The `idx_lookup` vector itself (6.46M entries) is built once, which is fine, but the per-row key construction is the bottleneck.

2. **The lookup is year-redundant.** Neighbors are purely spatial — they don't change across years. Yet the code re-discovers "which rows belong to neighbor cell X in year Y" independently for every row. Since every cell appears in every year, the neighbor *row-index offsets* are **structurally identical across years** and can be computed once from the spatial topology alone.

3. **`compute_neighbor_stats`** is called 5 times (once per variable), each time re-traversing the 6.46M-entry lookup list. This is comparatively cheap versus the build step, but a vectorized/matrix approach eliminates the `lapply` entirely.

### Summary of Redundancies

| Layer | What's repeated | Scale |
|---|---|---|
| String construction | `paste(id, year)` per row | 6.46M × ~4 neighbors |
| Hash lookup | Named-vector indexing per row | 6.46M × ~4 neighbors |
| Year dimension | Same spatial topology re-resolved per year | 28× redundant |
| Variable loop | `lapply` over 6.46M list elements per variable | 5× |

---

## Optimization Strategy

### Key Insight: Separate the Spatial Topology from the Temporal Dimension

Since every cell appears in every year, and neighbors are purely spatial:

1. **Build a spatial neighbor matrix once** — a sparse matrix or a simple integer-index list mapping each *cell* (not cell-year) to its neighbor *cells*. This is just `rook_neighbors_unique` translated to integer indices. Cost: O(344K cells).

2. **Sort/index the panel so that rows for the same cell are contiguous or easily addressable by cell-index and year-offset.** If the data is sorted by `(id, year)`, then the row for cell `c` in year `y` is simply `(c_index - 1) * 28 + (y - 1992) + 1`. This replaces all hash lookups with arithmetic.

3. **Vectorize the neighbor statistics** using matrix operations. Reshape each variable to a `344,208 × 28` matrix (cells × years). For each cell, its neighbor values in any year are just the neighbor-row slices of that matrix column. We can compute max/min/mean across neighbors using sparse-matrix multiplication (for mean) and row-wise operations.

4. **Use a sparse neighbor matrix `W`** and matrix multiplication `W %*% X` to get neighbor sums in one shot, then divide by neighbor counts for means. For min/max, iterate over cells (not cell-years) — reducing the loop from 6.46M to 344K, a **~18.7× speedup** on the inner loop alone, with each iteration doing simple integer-indexed vector subsetting instead of hash lookups.

### Expected Speedup

| Component | Before | After | Factor |
|---|---|---|---|
| Lookup build | ~6.46M hash lookups | Eliminated (arithmetic indexing) | ∞ |
| Neighbor stats (mean) | 6.46M × 5 `lapply` | 5 sparse matmuls (344K × 344K sparse) × 28 cols | ~100–500× |
| Neighbor stats (min/max) | 6.46M × 5 | 344K × 5 loops over integer vectors | ~18× |
| Total estimated time | 86+ hours | **~5–15 minutes** | ~350–1000× |

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor-feature construction
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact same numerical output columns; trained RF model untouched
# =============================================================================

library(Matrix)  # for sparse matrix operations

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------
  # 1. Establish cell-to-index and row-addressing scheme
  # -------------------------------------------------------------------
  # Ensure data is sorted by (id, year) so we can use arithmetic indexing.
  # If not already sorted, sort and record the original order to restore later.

  cell_data$.orig_order <- seq_len(nrow(cell_data))
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

  unique_ids   <- sort(unique(cell_data$id))
  unique_years <- sort(unique(cell_data$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)

  stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

  # Map cell id -> integer index (1..n_cells)
  id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))

  # Map year -> integer offset (1..n_years)
  year_to_offset <- setNames(seq_along(unique_years), as.character(unique_years))

  # Row address function: cell index c (1-based), year offset t (1-based)
  # row = (c - 1) * n_years + t
  # This works because data is sorted by (id, year).

  cat("Panel dimensions:", n_cells, "cells ×", n_years, "years =",
      n_cells * n_years, "rows\n")

  # -------------------------------------------------------------------
  # 2. Build spatial neighbor list in terms of cell indices
  #    (translate from id_order / nb object to integer cell indices)
  # -------------------------------------------------------------------
  # id_order maps position-in-nb-object -> cell id
  # We need: for each cell index (in unique_ids order), its neighbor cell indices

  id_order_to_idx <- id_to_idx[as.character(id_order)]

  # rook_neighbors_unique is an nb object: list of integer vectors
  # rook_neighbors_unique[[k]] gives the positions (in id_order) of neighbors of id_order[k]

  # Build neighbor list indexed by our cell index
  neighbor_list <- vector("list", n_cells)

  for (k in seq_along(id_order)) {
    cell_idx <- id_order_to_idx[k]
    nb_positions <- rook_neighbors_unique[[k]]
    # nb objects use 0 to indicate no neighbors
    if (length(nb_positions) == 1 && nb_positions[1] == 0L) {
      neighbor_list[[cell_idx]] <- integer(0)
    } else {
      neighbor_list[[cell_idx]] <- as.integer(id_order_to_idx[nb_positions])
    }
  }

  # -------------------------------------------------------------------
  # 3. Build sparse neighbor matrix W (n_cells × n_cells)
  #    W[i,j] = 1 if j is a neighbor of i
  # -------------------------------------------------------------------
  # This is used for computing neighbor means via matrix multiplication.

  n_entries <- sum(vapply(neighbor_list, length, integer(1)))
  cat("Building sparse W matrix:", n_cells, "×", n_cells,
      "with", n_entries, "non-zero entries\n")

  row_i <- integer(n_entries)
  col_j <- integer(n_entries)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- neighbor_list[[i]]
    len <- length(nb)
    if (len > 0L) {
      row_i[pos:(pos + len - 1L)] <- i
      col_j[pos:(pos + len - 1L)] <- nb
      pos <- pos + len
    }
  }

  W <- sparseMatrix(i = row_i, j = col_j, x = 1.0,
                    dims = c(n_cells, n_cells))

  # Neighbor count per cell (used for mean calculation)
  neighbor_count <- as.numeric(W %*% rep(1.0, n_cells))  # length n_cells

  rm(row_i, col_j)  # free memory

  # -------------------------------------------------------------------
  # 4. For each source variable, compute neighbor max, min, mean
  #    using matrix operations + vectorized cell-level loops
  # -------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")

    vals <- cell_data[[var_name]]

    # Reshape to matrix: rows = cells, cols = years
    # Row c, col t corresponds to original row (c-1)*n_years + t
    V <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = TRUE)

    # --- Neighbor mean via sparse matrix multiplication ---
    # For each year (column), compute W %*% V[,t] = sum of neighbor values
    # Then divide by neighbor_count.
    # Handle NAs: we need mean of non-NA neighbor values.

    # To correctly handle NAs:
    # - Replace NA with 0 for sum computation
    # - Count non-NA neighbors per cell-year
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0

    # Indicator of non-NA
    V_valid <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)

    # Sparse matmul: W %*% V_nona gives neighbor sums (treating NA as 0)
    # W %*% V_valid gives count of non-NA neighbors per cell-year
    neighbor_sum   <- as.matrix(W %*% V_nona)    # n_cells × n_years
    neighbor_nvalid <- as.matrix(W %*% V_valid)   # n_cells × n_years

    neighbor_mean_mat <- neighbor_sum / neighbor_nvalid
    neighbor_mean_mat[neighbor_nvalid == 0] <- NA

    # --- Neighbor min and max: must iterate over cells (344K, not 6.46M) ---
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (ci in seq_len(n_cells)) {
      nb <- neighbor_list[[ci]]
      if (length(nb) == 0L) next
      # Extract the sub-matrix for all neighbors across all years
      # nb_vals: length(nb) × n_years matrix
      nb_vals <- V[nb, , drop = FALSE]

      if (length(nb) == 1L) {
        # Single neighbor: max = min = that value (may be NA)
        neighbor_max_mat[ci, ] <- nb_vals[1L, ]
        neighbor_min_mat[ci, ] <- nb_vals[1L, ]
      } else {
        # Column-wise max and min ignoring NAs
        # Use matrixStats if available for speed, otherwise base R
        for (t in seq_len(n_years)) {
          col_vals <- nb_vals[, t]
          col_vals <- col_vals[!is.na(col_vals)]
          if (length(col_vals) > 0L) {
            neighbor_max_mat[ci, t] <- max(col_vals)
            neighbor_min_mat[ci, t] <- min(col_vals)
          }
        }
      }

      # Progress indicator (every 50K cells)
      if (ci %% 50000L == 0L) {
        cat("  ", var_name, ": processed", ci, "/", n_cells, "cells\n")
      }
    }

    # --- Flatten back to panel vector (row-major: cell × year) ---
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    cell_data[[max_col_name]]  <- as.vector(t(neighbor_max_mat))
    cell_data[[min_col_name]]  <- as.vector(t(neighbor_min_mat))
    cell_data[[mean_col_name]] <- as.vector(t(neighbor_mean_mat))

    # Wait — the flattening must match the sorted row order.
    # Data is sorted by (id, year). Row order is:
    #   cell_1-year_1, cell_1-year_2, ..., cell_1-year_28,
    #   cell_2-year_1, ...
    # Matrix is stored row-major as: V[c, t] -> row (c-1)*n_years + t
    # as.vector(t(M)) reads M row by row, which gives exactly this order. ✓

    rm(V, V_nona, V_valid, neighbor_sum, neighbor_nvalid,
       neighbor_mean_mat, neighbor_max_mat, neighbor_min_mat)
  }

  # -------------------------------------------------------------------
  # 5. Restore original row order
  # -------------------------------------------------------------------
  cell_data <- cell_data[order(cell_data$.orig_order), ]
  cell_data$.orig_order <- NULL

  return(cell_data)
}


# =============================================================================
# Even faster min/max using matrixStats (if available)
# =============================================================================
# The inner loop over cells with the nested year loop can be accelerated
# significantly with matrixStats::colMaxs / colMins.
# Drop-in replacement for the min/max block:

build_neighbor_features_optimized_fast <- function(cell_data,
                                                    id_order,
                                                    rook_neighbors_unique,
                                                    neighbor_source_vars) {

  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    message("matrixStats not available; falling back to base R version.")
    return(build_neighbor_features_optimized(cell_data, id_order,
                                              rook_neighbors_unique,
                                              neighbor_source_vars))
  }

  cell_data$.orig_order <- seq_len(nrow(cell_data))
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

  unique_ids   <- sort(unique(cell_data$id))
  unique_years <- sort(unique(cell_data$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)

  stopifnot(nrow(cell_data) == n_cells * n_years)

  id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))
  id_order_to_idx <- id_to_idx[as.character(id_order)]

  # Build neighbor list
  neighbor_list <- vector("list", n_cells)
  for (k in seq_along(id_order)) {
    cell_idx <- id_order_to_idx[k]
    nb_positions <- rook_neighbors_unique[[k]]
    if (length(nb_positions) == 1 && nb_positions[1] == 0L) {
      neighbor_list[[cell_idx]] <- integer(0)
    } else {
      neighbor_list[[cell_idx]] <- as.integer(id_order_to_idx[nb_positions])
    }
  }

  # Sparse matrix W
  n_entries <- sum(vapply(neighbor_list, length, integer(1)))
  row_i <- integer(n_entries)
  col_j <- integer(n_entries)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- neighbor_list[[i]]
    len <- length(nb)
    if (len > 0L) {
      row_i[pos:(pos + len - 1L)] <- i
      col_j[pos:(pos + len - 1L)] <- nb
      pos <- pos + len
    }
  }
  W <- sparseMatrix(i = row_i, j = col_j, x = 1.0,
                    dims = c(n_cells, n_cells))
  rm(row_i, col_j)

  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")

    vals <- cell_data[[var_name]]
    V <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = TRUE)

    # --- Mean (sparse matmul, NA-safe) ---
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    V_valid <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)

    neighbor_sum    <- as.matrix(W %*% V_nona)
    neighbor_nvalid <- as.matrix(W %*% V_valid)
    neighbor_mean_mat <- neighbor_sum / neighbor_nvalid
    neighbor_mean_mat[neighbor_nvalid == 0] <- NA

    # --- Min / Max (cell-level loop with matrixStats) ---
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (ci in seq_len(n_cells)) {
      nb <- neighbor_list[[ci]]
      if (length(nb) == 0L) next
      nb_vals <- V[nb, , drop = FALSE]  # len(nb) × n_years

      if (length(nb) == 1L) {
        neighbor_max_mat[ci, ] <- nb_vals[1L, ]
        neighbor_min_mat[ci, ] <- nb_vals[1L, ]
      } else {
        neighbor_max_mat[ci, ] <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
        neighbor_min_mat[ci, ] <- matrixStats::colMins(nb_vals, na.rm = TRUE)
      }

      if (ci %% 100000L == 0L) cat("  ", var_name, ":", ci, "/", n_cells, "\n")
    }

    # Fix -Inf/Inf from all-NA columns
    neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA
    neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA

    # Flatten and assign
    cell_data[[paste0("neighbor_max_", var_name)]]  <- as.vector(t(neighbor_max_mat))
    cell_data[[paste0("neighbor_min_", var_name)]]   <- as.vector(t(neighbor_min_mat))
    cell_data[[paste0("neighbor_mean_", var_name)]] <- as.vector(t(neighbor_mean_mat))

    rm(V, V_nona, V_valid, neighbor_sum, neighbor_nvalid,
       neighbor_mean_mat, neighbor_max_mat, neighbor_min_mat)
    gc()
  }

  cell_data <- cell_data[order(cell_data$.orig_order), ]
  cell_data$.orig_order <- NULL
  return(cell_data)
}


# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================
#
# # Original code:
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Optimized replacement (single call):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_optimized_fast(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched — only feature columns are
# # reconstructed with identical numerical values. Prediction proceeds as before:
# # predictions <- predict(rf_model, cell_data[, predictor_cols])
```

---

## Why This Works and What Changed

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string `paste` + hash probes | Eliminated entirely — arithmetic row addressing via sorted panel |
| **Neighbor discovery** | Redone per cell-year (6.46M times) | Done once per cell (344K times), reused across all 28 years |
| **Mean computation** | `lapply` over 6.46M entries, 5 variables | 5 sparse matrix multiplications (`W %*% V`), each ~O(n_entries × n_years) ≈ 38M FLOPs, fully vectorized in C |
| **Min/Max computation** | `lapply` over 6.46M entries | Loop over 344K cells with `matrixStats::colMaxs/colMins` (vectorized C per cell) |
| **Memory** | 6.46M-element list of integer vectors (~500MB+) | ~344K-entry neighbor list + one `n_cells × n_years` matrix at a time (~77MB), well within 16GB |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** |
| **Numerical output** | max/min/mean of non-NA neighbor values per cell-year | Identical — same neighbor sets, same aggregation functions, same NA handling |
| **RF model** | N/A | Completely untouched — only the feature columns are recomputed with identical values |