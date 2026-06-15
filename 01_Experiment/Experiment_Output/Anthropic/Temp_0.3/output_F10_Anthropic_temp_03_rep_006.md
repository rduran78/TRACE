 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries** by iterating row-by-row through the entire panel. For each of the 6.46M rows, it performs string-pasting, hash lookups, and NA filtering. This is O(rows × avg_neighbors) with heavy per-element overhead from R's interpreted loop and character key hashing.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** per variable, subsetting a numeric vector by index vectors stored in a list. With 5 variables, that's ~32.3M list traversals, each involving allocation of small numeric vectors and triple-pass (max/min/mean) computation.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors don't change across years — cell *i*'s neighbors are the same cells in 1992 as in 2019. The current code fuses topology with temporal indexing, preventing reuse and forcing string-key lookups for every cell-year.

**Root cause:** The design treats a **separable** problem (spatial topology × temporal panel) as a **joint** problem, inflating the work by a factor of 28 (years). The actual spatial graph has only 344,208 nodes and ~1.37M edges. All neighbor aggregation can be expressed as **sparse matrix–dense matrix multiplication** (and element-wise operations), which is highly optimized in compiled code.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208, ~1.37M non-zero entries). This is the graph topology in CSC/CSR format.

2. **Reshape each variable into a dense matrix** of dimension (344,208 cells × 28 years). This separates spatial and temporal dimensions.

3. **Compute neighbor aggregates via sparse matrix operations:**
   - **Neighbor sum** = `A %*% X` (sparse × dense, compiled CHOLMOD/CSparse code)
   - **Neighbor count** = `A %*% (!is.na(X))` (to handle missing values correctly)
   - **Neighbor mean** = sum / count
   - **Neighbor max and min** require a custom approach since sparse matrix algebra doesn't directly support element-wise max/min. We use a **row-wise iteration over the sparse matrix** in C++ via `Rcpp`, or — staying in pure R — we iterate over the 344,208 *cells* (not 6.46M cell-years) and vectorize across years.

4. **Flatten back** to the original long-format data frame, attach the 15 new columns (5 vars × 3 stats), and predict with the existing Random Forest.

**Expected speedup:** The sparse matrix multiply for sum/count handles all 28 years simultaneously in compiled code — this alone replaces ~70% of the work. The max/min loop runs over 344K cells (not 6.46M rows), with each iteration vectorized across 28 years. Estimated wall time: **2–10 minutes** instead of 86+ hours.

## Working R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original max/min/mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)   # sparse matrices
library(data.table)  # fast reshaping and joining

# ---- 0. Ensure cell_data is a data.table with original row ordering --------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}
# Preserve original row order for final reassembly
cell_data[, .row_order := .I]

# ---- 1. Build spatial topology ONCE as a sparse logical adjacency matrix ---
# id_order: vector of length 344,208 giving the cell id for each nb index
# rook_neighbors_unique: nb object (list of length 344,208)

n_cells <- length(id_order)
stopifnot(n_cells == length(rook_neighbors_unique))

# Build COO triplets from the nb object
# nb objects use 0 to indicate no neighbors; filter those out
from_idx <- rep(seq_len(n_cells),
                times = vapply(rook_neighbors_unique, function(x) {
                  sum(x > 0L)
                }, integer(1)))
to_idx   <- unlist(lapply(rook_neighbors_unique, function(x) x[x > 0L]),
                   use.names = FALSE)

# Sparse adjacency matrix: A[i,j] = 1 means j is a rook neighbor of i
# Dimensions: n_cells x n_cells
# This means row i contains 1s in columns corresponding to neighbors of cell i
A <- sparseMatrix(
  i    = from_idx,
  j    = to_idx,
  x    = 1,
  dims = c(n_cells, n_cells),
  repr = "C"   # CSC format, efficient for %*%
)

rm(from_idx, to_idx)
gc()

cat("Adjacency matrix:", n_cells, "x", n_cells,
    "with", nnzero(A), "non-zero entries\n")

# ---- 2. Create cell-index and year-index mappings --------------------------
# Map cell ids to matrix row indices (1..n_cells)
cell_id_to_row <- setNames(seq_len(n_cells), as.character(id_order))

# Sorted unique years
years_sorted <- sort(unique(cell_data$year))
n_years      <- length(years_sorted)
year_to_col  <- setNames(seq_len(n_years), as.character(years_sorted))

cat("Panel:", n_cells, "cells x", n_years, "years =",
    n_cells * n_years, "potential cell-years\n")

# Map each row of cell_data to (cell_row_index, year_col_index)
cell_data[, .cell_row := cell_id_to_row[as.character(id)]]
cell_data[, .year_col := year_to_col[as.character(year)]]

# ---- 3. Function: reshape a variable to dense matrix (cells x years) -------
var_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$.cell_row, dt$.year_col)] <- dt[[var_name]]
  mat
}

# ---- 4. Function: compute neighbor max, min, mean for one variable ---------
#    Returns three matrices, each n_cells x n_years.
#    Numerically equivalent to the original per-row computation.

compute_neighbor_stats_sparse <- function(A, X) {
  # A: n_cells x n_cells sparse adjacency (CSC)
  # X: n_cells x n_years dense matrix (may contain NAs)
  
  n_cells <- nrow(X)
  n_years <- ncol(X)
  
  # --- Mean via sparse matrix multiply ---
  # Replace NAs with 0 for summation, track non-NA counts
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  
  X_notna <- matrix(1, nrow = n_cells, ncol = n_years)
  X_notna[is.na(X)] <- 0
  
  # Neighbor sums and counts (compiled sparse %*% dense)
  neighbor_sum   <- as.matrix(A %*% X_nona)    # n_cells x n_years
  neighbor_count <- as.matrix(A %*% X_notna)    # n_cells x n_years
  
  # Mean: sum / count, NA where count == 0
  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  rm(X_nona, X_notna, neighbor_sum)
  gc()
  
  # --- Max and Min via row-wise iteration over sparse structure ---
  # We iterate over 344K cells (not 6.46M cell-years).
  # For each cell, we gather neighbor rows from X and compute
  # column-wise max and min (vectorized across years).
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Convert to dgRMatrix (CSR) for efficient row access
  A_csr <- as(A, "RsparseMatrix")
  
  # Extract CSR components
  # For dgRMatrix: @p is row pointer (length n_cells+1), @j is column indices (0-based)
  row_ptr <- A_csr@p    # length n_cells + 1
  col_idx <- A_csr@j    # 0-based column indices
  
  for (i in seq_len(n_cells)) {
    start <- row_ptr[i] + 1L    # convert to 1-based
    end   <- row_ptr[i + 1L]
    
    if (end < start) next  # no neighbors
    
    nb_indices <- col_idx[start:end] + 1L  # 1-based neighbor row indices
    
    if (length(nb_indices) == 1L) {
      # Single neighbor: just copy its values (respecting NAs)
      nb_row <- X[nb_indices, , drop = FALSE]
      neighbor_max[i, ] <- nb_row
      neighbor_min[i, ] <- nb_row
    } else {
      # Multiple neighbors: column-wise max/min
      nb_block <- X[nb_indices, , drop = FALSE]  # small matrix: ~4 rows x 28 cols
      
      # Use colMins/colMaxs-style via apply or matrixStats if available
      # For a block of ~4 rows x 28 cols, apply is fine
      neighbor_max[i, ] <- apply(nb_block, 2, max, na.rm = TRUE)
      neighbor_min[i, ] <- apply(nb_block, 2, min, na.rm = TRUE)
    }
  }
  
  # apply(x, 2, max, na.rm=TRUE) returns -Inf for all-NA columns; fix to NA
  neighbor_max[is.infinite(neighbor_max)] <- NA_real_
  neighbor_min[is.infinite(neighbor_min)] <- NA_real_
  
  list(max = neighbor_max, min = neighbor_min, mean = neighbor_mean)
}

# ---- 5. Main loop: compute and attach neighbor features --------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  t0 <- proc.time()
  
  # Reshape to matrix
  X <- var_to_matrix(cell_data, var_name, n_cells, n_years)
  
  # Compute neighbor stats
  stats <- compute_neighbor_stats_sparse(A, X)
  rm(X)
  
  # Flatten back to cell_data's row order using .cell_row and .year_col
  idx_mat <- cbind(cell_data$.cell_row, cell_data$.year_col)
  
  max_col_name  <- paste0("neighbor_max_", var_name)
  min_col_name  <- paste0("neighbor_min_", var_name)
  mean_col_name <- paste0("neighbor_mean_", var_name)
  
  set(cell_data, j = max_col_name,  value = stats$max[idx_mat])
  set(cell_data, j = min_col_name,  value = stats$min[idx_mat])
  set(cell_data, j = mean_col_name, value = stats$mean[idx_mat])
  
  rm(stats)
  gc()
  
  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# ---- 6. Clean up helper columns -------------------------------------------
cell_data[, c(".cell_row", ".year_col", ".row_order") := NULL]

# ---- 7. Predict with the pre-trained Random Forest (unchanged) -------------
# The RF model expects the same column names as the training data.
# The 15 new neighbor columns (5 vars x 3 stats) are now attached.
# Predict exactly as before:

# predictions <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete.\n")
cat("Dataset dimensions:", nrow(cell_data), "rows x", ncol(cell_data), "cols\n")
```

## Further Optimization: Rcpp for Max/Min (Optional)

The `for` loop over 344K cells with `apply` on tiny blocks is the remaining bottleneck (~80% of remaining time). If it's still too slow, this Rcpp function eliminates all R-level overhead:

```r
# Optional: Rcpp acceleration for max/min
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_csr(IntegerVector row_ptr,
                         IntegerVector col_idx,
                         NumericMatrix X) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix nmax(n, T);
  NumericMatrix nmin(n, T);
  
  // Initialize to NA
  std::fill(nmax.begin(), nmax.end(), NA_REAL);
  std::fill(nmin.begin(), nmin.end(), NA_REAL);
  
  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    if (start >= end) continue;  // no neighbors
    
    for (int t = 0; t < T; t++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      bool any_valid = false;
      
      for (int k = start; k < end; k++) {
        double val = X(col_idx[k], t);
        if (!R_IsNA(val)) {
          any_valid = true;
          if (val > mx) mx = val;
          if (val < mn) mn = val;
        }
      }
      
      if (any_valid) {
        nmax(i, t) = mx;
        nmin(i, t) = mn;
      }
    }
  }
  
  return List::create(Named("max") = nmax, Named("min") = nmin);
}
')

# Replace the R for-loop in compute_neighbor_stats_sparse with:
# A_csr <- as(A, "RsparseMatrix")
# maxmin <- neighbor_maxmin_csr(A_csr@p, A_csr@j, X)
# neighbor_max <- maxmin$max
# neighbor_min <- maxmin$min
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology representation** | 6.46M-element list of index vectors | 344K × 344K sparse matrix (1.37M entries) |
| **Key lookup** | String paste + hash per cell-year | Integer matrix indexing |
| **Mean computation** | R loop over 6.46M elements | Sparse `A %*% X` (compiled BLAS) |
| **Max/Min computation** | R loop over 6.46M elements | Loop over 344K cells, vectorized over 28 years (or Rcpp) |
| **Memory** | ~6.46M list entries + string keys | ~344K × 28 dense matrices + sparse matrix |
| **Estimated time** | 86+ hours | **2–10 minutes** (pure R) / **< 1 minute** (with Rcpp) |
| **Numerical equivalence** | Baseline | Identical — same max, min, mean per cell-year |
| **RF model** | Unchanged | Unchanged — no retraining |