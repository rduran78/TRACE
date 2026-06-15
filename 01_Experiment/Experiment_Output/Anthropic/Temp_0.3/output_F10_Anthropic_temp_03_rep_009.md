 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list elements**, each via an anonymous function call with string concatenation (`paste`), named-vector lookups, and NA filtering. This is an O(n) loop over millions of rows with expensive string operations at each step.

2. **`compute_neighbor_stats` iterates over ~6.46 million list elements** per variable, calling `max`, `min`, `mean` inside an `lapply`. With 5 variables, that's ~32.3 million R-level function invocations with per-element allocation.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Each cell has the same rook neighbors every year. The `paste(id, year)` key construction entangles topology with time, preventing vectorized exploitation of the panel's regular structure.

**Root cause:** The design treats the problem as a flat row-level operation instead of exploiting the fact that the adjacency structure is **static across years** and the panel is **balanced** (every cell appears in every year). This means neighbor aggregation can be expressed as a **sparse matrix–dense matrix multiplication**, which is O(nnz) and runs in optimized C/Fortran code.

---

## Optimization Strategy

### Key Insight
If `A` is the `N_cells × N_cells` sparse adjacency matrix (rook neighbors), and `X` is an `N_cells × N_years` matrix of a variable (one column per year), then:

- **Neighbor sum** = `A %*% X` (sparse matrix multiply, O(nnz × T))
- **Neighbor count** = `A %*% (non-NA indicator matrix)` (same cost)
- **Neighbor mean** = sum / count
- **Neighbor max/min** require a grouped operation, but can be vectorized via the CSR representation of `A`

For **mean**, sparse matrix multiplication gives us exact numerical equivalence. For **max** and **min**, we iterate over the CSR row pointers in C++ via `Rcpp`, which is O(nnz × T) with no R-level per-element overhead.

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Lookup build | O(R) string ops, R ≈ 6.46M | O(N) integer ops, N = 344K (once) |
| Mean (per var) | O(R) R-level loops | O(nnz × T) BLAS sparse multiply |
| Max/Min (per var) | O(R) R-level loops | O(nnz × T) compiled C++ |
| Total R-level calls | ~32M+ | ~5 (one per variable) |
| Estimated time | 86+ hours | **Minutes** |

---

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats.
# Requires: Matrix, Rcpp, data.table
# ==============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix from spdep nb object (ONCE)
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_object, n_cells) {
  # nb_object: list of length n_cells, each element is integer vector of neighbor indices
  # Builds a sparse CSR-compatible matrix (dgRMatrix) via COO -> dgCMatrix -> dgRMatrix
  
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n_cells)) {
    nbrs <- nb_object[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from <- c(from, rep.int(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }
  
  A <- sparseMatrix(
    i    = from,
    j    = to,
    x    = rep.int(1, length(from)),
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format initially
  )
  
  return(A)
}

# --------------------------------------------------------------------------
# STEP 2: Rcpp function for sparse row-wise max and min over a dense matrix
#          Operates on CSR representation for cache-friendly row traversal.
# --------------------------------------------------------------------------
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_row_maxmin(IntegerVector row_ptr,    // length n_rows + 1, 0-based
                       IntegerVector col_idx,    // length nnz, 0-based
                       NumericMatrix X,          // n_cells x n_years
                       int n_rows) {
  int n_cols = X.ncol();
  NumericMatrix out_max(n_rows, n_cols);
  NumericMatrix out_min(n_rows, n_cols);
  
  // Initialize to NA
  double na_val = NA_REAL;
  std::fill(out_max.begin(), out_max.end(), na_val);
  std::fill(out_min.begin(), out_min.end(), na_val);
  
  for (int i = 0; i < n_rows; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    
    if (start == end) continue;  // no neighbors, stays NA
    
    for (int t = 0; t < n_cols; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid_count = 0;
      
      for (int jj = start; jj < end; jj++) {
        int neighbor = col_idx[jj];
        double val = X(neighbor, t);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid_count++;
        }
      }
      
      if (valid_count > 0) {
        out_max(i, t) = cur_max;
        out_min(i, t) = cur_min;
      }
      // else stays NA
    }
  }
  
  return List::create(Named("max_mat") = out_max,
                      Named("min_mat") = out_min);
}
')

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor mean via sparse matrix multiplication
#          Handles NAs correctly: mean = sum_of_non_NA / count_of_non_NA
# --------------------------------------------------------------------------
compute_neighbor_mean_sparse <- function(A, X_mat) {
  # X_mat: n_cells x n_years, may contain NAs
  # Replace NAs with 0 for summation, track non-NA indicator
  
  X_clean       <- X_mat
  na_mask       <- is.na(X_mat)
  X_clean[na_mask] <- 0
  
  indicator     <- matrix(1, nrow = nrow(X_mat), ncol = ncol(X_mat))
  indicator[na_mask] <- 0
  
  # Sparse multiply: A (n_cells x n_cells) %*% X_clean (n_cells x n_years)
  neighbor_sum   <- A %*% X_clean
  neighbor_count <- A %*% indicator
  
  # Convert to dense
  neighbor_sum   <- as.matrix(neighbor_sum)
  neighbor_count <- as.matrix(neighbor_count)
  
  # mean = sum / count; where count == 0, result is NA
  result <- neighbor_sum / neighbor_count
  result[neighbor_count == 0] <- NA_real_
  
  return(result)
}

# --------------------------------------------------------------------------
# STEP 4: Master function — reshape, aggregate, reshape back
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, 
                                           id_order, 
                                           rook_neighbors_unique,
                                           neighbor_source_vars) {
  
  n_cells <- length(id_order)
  
  cat("Building sparse adjacency matrix...\n")
  A_csc <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  
  # Convert to CSR (dgRMatrix) for the C++ max/min kernel
  A_csr <- as(A_csc, "RsparseMatrix")
  
  # Build cell-id to matrix-row mapping
  id_to_row <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Convert to data.table for fast reshaping
  dt <- as.data.table(cell_data)
  
  # Get sorted unique years for consistent column ordering
  years_sorted <- sort(unique(dt$year))
  n_years      <- length(years_sorted)
  year_to_col  <- setNames(seq_len(n_years), as.character(years_sorted))
  
  # Map each row to its (matrix_row, matrix_col) position
  dt[, mat_row := id_to_row[as.character(id)]]
  dt[, mat_col := year_to_col[as.character(year)]]
  
  # Extract CSR components for C++ (0-based indexing)
  # dgRMatrix slots: @p (row pointers), @j (column indices), both 0-based
  row_ptr <- A_csr@p        # integer, length n_cells + 1, 0-based
  col_idx <- A_csr@j        # integer, length nnz, 0-based
  
  cat(sprintf("Adjacency: %d cells, %d directed edges, %d years\n",
              n_cells, length(col_idx), n_years))
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # --- Reshape variable into n_cells x n_years dense matrix ---
    X_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    vals  <- dt[[var_name]]
    rows  <- dt$mat_row
    cols  <- dt$mat_col
    X_mat[cbind(rows, cols)] <- vals
    
    # --- Compute neighbor mean via sparse matmul ---
    mean_mat <- compute_neighbor_mean_sparse(A_csc, X_mat)
    
    # --- Compute neighbor max and min via C++ CSR traversal ---
    maxmin <- sparse_row_maxmin(row_ptr, col_idx, X_mat, n_cells)
    max_mat <- maxmin$max_mat
    min_mat <- maxmin$min_mat
    
    # --- Map results back to the original row order ---
    idx <- cbind(rows, cols)
    
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col_name)  := max_mat[idx]]
    dt[, (min_col_name)  := min_mat[idx]]
    dt[, (mean_col_name) := mean_mat[idx]]
    
    cat(sprintf("  Done: %s, %s, %s\n", max_col_name, min_col_name, mean_col_name))
  }
  
  # Remove helper columns
  dt[, c("mat_row", "mat_col") := NULL]
  
  return(as.data.frame(dt))
}

# ==========================================================================
# EXECUTION
# ==========================================================================

# --- Compute all neighbor features ---
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# --- Apply the pre-trained Random Forest (NO retraining) ---
# The model object (e.g., rf_model) is already in memory or loaded from disk.
# Predict using the enriched cell_data which now has all ~110 predictor columns.
cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

---

## Numerical Equivalence Verification

Run this once on a small subset to confirm the optimized code produces identical results:

```r
# --------------------------------------------------------------------------
# VERIFICATION: Compare original vs optimized on a sample
# --------------------------------------------------------------------------
verify_equivalence <- function(cell_data_original, cell_data_optimized, 
                                neighbor_source_vars, n_check = 1000) {
  set.seed(42)
  check_rows <- sample(nrow(cell_data_original), min(n_check, nrow(cell_data_original)))
  
  all_ok <- TRUE
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("neighbor_", stat, "_", var_name)
      orig <- cell_data_original[[col]][check_rows]
      optm <- cell_data_optimized[[col]][check_rows]
      
      both_na <- is.na(orig) & is.na(optm)
      both_val <- !is.na(orig) & !is.na(optm)
      
      na_match <- all(both_na == (is.na(orig) | is.na(optm)))  
      # More precisely:
      na_mismatch <- sum(is.na(orig) != is.na(optm))
      val_match   <- all(abs(orig[both_val] - optm[both_val]) < 1e-10)
      
      if (na_mismatch > 0 || !val_match) {
        cat(sprintf("MISMATCH: %s — NA mismatches: %d, value match: %s\n",
                    col, na_mismatch, val_match))
        all_ok <- FALSE
      }
    }
  }
  
  if (all_ok) cat("All checks passed: numerical equivalence confirmed.\n")
  return(all_ok)
}
```

---

## Performance Summary

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| Topology construction | ~6.46M string ops | ~1.37M integer COO entries → sparse matrix | ~100× |
| Mean aggregation (per var) | 6.46M R `lapply` calls | 1 sparse matmul (CHOLMOD/BLAS) | ~500× |
| Max/Min aggregation (per var) | 6.46M R `lapply` calls | 1 C++ CSR pass | ~200× |
| Memory pattern | ~6.46M R list elements + string keys | Dense matrix (344K × 28 ≈ 77MB per var) + sparse matrix (~22MB) | ~3–5× less |
| **Total estimated runtime** | **86+ hours** | **5–15 minutes** | **~500×** |

The trained Random Forest model is never modified or retrained — it is applied as-is via `predict()` on the enriched dataset.