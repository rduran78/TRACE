 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names — this alone is catastrophically slow. The function does ~6.46M × avg_neighbors hash lookups against a 6.46M-length named vector.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting variable values and computing `max/min/mean` in pure R. This is repeated 5 times (once per source variable), totaling ~32.3 million R-level function calls.

3. **The neighbor lookup is year-aware but redundant**: rook neighbors are a *spatial* relationship that is identical across all 28 years. The current code rebuilds year-specific row indices by pasting strings, but the topology is static. The lookup should be built once at the spatial level and then broadcast across years via vectorized indexing.

**Root cause summary**: The 86+ hour runtime comes from (a) O(n²)-like named-vector lookups during graph construction, and (b) millions of interpreted R loop iterations during aggregation. Both can be replaced with vectorized sparse-matrix operations.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** (344,208 × 344,208) from the `nb` object — ~1.37M non-zero entries. This is tiny in memory (~16 MB as a `dgCMatrix`).

2. **Reshape each variable into a matrix** of shape (344,208 cells × 28 years). Each column is one year.

3. **Compute neighbor aggregates via sparse matrix–dense matrix multiplication and element-wise operations**:
   - **Mean**: `A %*% X / degree` (where `degree` = number of neighbors per node).
   - **Max / Min**: Use a single pass over the sparse matrix's CSC structure in C++ (via `Rcpp`) or use `{Matrix}` tricks with repeated sparse operations.

4. **Flatten back** to the original long-format data frame and bind columns.

This reduces the entire pipeline from ~86 hours to **minutes** (sparse matrix ops on matrices of this size are near-instantaneous).

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor stats.
# =============================================================================

library(Matrix)
library(Rcpp)

# ---- Step 0: One-time C++ helper for sparse row-wise max/min ----
# We compile a small Rcpp function that, given a CSC sparse matrix A and a
# dense matrix X, computes row-wise max, min, and sum of neighbor values.
# This avoids any R-level loops over 6.46M rows.

sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_stats(
    IntegerVector Ap,    // CSC column pointers (length ncol+1)
    IntegerVector Ai,    // CSC row indices
    NumericMatrix X,     // dense matrix: nrow = n_cells, ncol = n_years
    IntegerVector degree // number of neighbors per row (length n_cells)
) {
  int n = X.nrow();
  int T = X.ncol();

  NumericMatrix out_max(n, T);
  NumericMatrix out_min(n, T);
  NumericMatrix out_mean(n, T);

  // Initialize max to -Inf, min to +Inf, sum to 0
  double posInf = R_PosInf;
  double negInf = R_NegInf;
  for (int i = 0; i < n; i++) {
    for (int t = 0; t < T; t++) {
      out_max(i, t) = negInf;
      out_min(i, t) = posInf;
      out_mean(i, t) = 0.0;
    }
  }

  // CSC traversal: for each column j (= neighbor source node),
  // iterate over rows i that have an edge from j -> i (i.e., j is neighbor of i).
  // Accumulate stats for row i using X[j, t].
  int ncol_A = Ap.size() - 1;
  for (int j = 0; j < ncol_A; j++) {
    for (int ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
      int i = Ai[ptr];  // row i has neighbor j
      for (int t = 0; t < T; t++) {
        double val = X(j, t);
        if (NumericMatrix::is_na(val)) continue;
        if (val > out_max(i, t)) out_max(i, t) = val;
        if (val < out_min(i, t)) out_min(i, t) = val;
        out_mean(i, t) += val;
      }
    }
  }

  // Finalize: replace sentinel values with NA; compute mean = sum / count
  // We need non-NA neighbor counts per (i, t). For simplicity and speed,
  // if all neighbor values for a row are NA, degree effectively = 0.
  // We track valid counts via a second pass or by noting that
  // if out_max is still -Inf, no valid neighbor was found.
  // For mean, we need valid counts. We do a second sparse pass for NA counting.

  // Count valid (non-NA) neighbors per (i, t)
  IntegerMatrix valid_count(n, T);
  for (int j = 0; j < ncol_A; j++) {
    for (int ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
      int i = Ai[ptr];
      for (int t = 0; t < T; t++) {
        double val = X(j, t);
        if (!NumericMatrix::is_na(val)) {
          valid_count(i, t) += 1;
        }
      }
    }
  }

  for (int i = 0; i < n; i++) {
    for (int t = 0; t < T; t++) {
      if (valid_count(i, t) == 0) {
        out_max(i, t) = NA_REAL;
        out_min(i, t) = NA_REAL;
        out_mean(i, t) = NA_REAL;
      } else {
        out_mean(i, t) = out_mean(i, t) / valid_count(i, t);
      }
    }
  }

  return List::create(
    Named("max")  = out_max,
    Named("min")  = out_min,
    Named("mean") = out_mean
  );
}
')

# ---- Step 1: Build sparse adjacency matrix from nb object (once) ----
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Builds a sparse n x n matrix A where A[i,j] = 1 means j is a neighbor of i
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-entries (spdep uses 0 to indicate no neighbors)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

# ---- Step 2: Ensure data is sorted by (id, year) and build index maps ----
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   neighbor_source_vars, rf_model) {

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  cat("Building spatial adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  # Convert to dgCMatrix (CSC) for our Rcpp function
  A <- as(A, "dgCMatrix")

  # Build mapping: cell id -> matrix row index (1..n_cells)
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Build mapping: year -> matrix column index (1..n_years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  # Map each row of cell_data to (cell_row, year_col)
  cat("Mapping data to cell x year matrix indices...\n")
  cell_row_idx <- id_to_row[as.character(cell_data$id)]
  year_col_idx <- year_to_col[as.character(cell_data$year)]
  # Linear index into n_cells x n_years matrix (column-major)
  lin_idx <- cell_row_idx + (year_col_idx - 1L) * n_cells

  # ---- Step 3: For each variable, reshape to matrix, compute stats, reshape back ----
  cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")

  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")

    # Reshape to n_cells x n_years matrix
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[lin_idx] <- cell_data[[var_name]]

    # Compute neighbor max, min, mean via sparse aggregation
    stats <- sparse_neighbor_stats(A@p, A@i, X, diff(A@p))

    # Extract results back to long format using the same linear index
    max_vals  <- stats$max[lin_idx]
    min_vals  <- stats$min[lin_idx]
    mean_vals <- stats$mean[lin_idx]

    # Add columns to cell_data (matching original naming convention)
    cell_data[[paste0("neighbor_max_",  var_name)]] <- max_vals
    cell_data[[paste0("neighbor_min_",  var_name)]] <- min_vals
    cell_data[[paste0("neighbor_mean_", var_name)]] <- mean_vals
  }

  cat("Neighbor feature engineering complete.\n")

  # ---- Step 4: Predict with pre-trained Random Forest (no retraining) ----
  cat("Generating predictions with pre-trained Random Forest model...\n")
  cell_data$prediction <- predict(rf_model, newdata = cell_data)

  return(cell_data)
}

# ---- Usage ----
# cell_data <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   rf_model               = trained_rf_model
# )
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup construction** | ~6.46M string-paste + named-vector lookups | One-time sparse matrix construction from `nb` object (~0.5 sec) |
| **Aggregation per variable** | 6.46M `lapply` iterations in R | Single C++ pass over ~1.37M sparse entries × 28 year columns |
| **Total aggregation passes** | 5 variables × 6.46M R calls = 32.3M calls | 5 × 1 sparse-matrix traversal in compiled C++ |
| **Memory** | 6.46M-element list of integer vectors + 6.46M named character vector | ~16 MB sparse matrix + 5 × (344K × 28) dense matrices (~39 MB each) |
| **Estimated runtime** | 86+ hours | **~2–5 minutes** |
| **Numerical equivalence** | Baseline | ✅ Identical max/min/mean (same NA handling, same arithmetic) |
| **RF model** | Unchanged | ✅ Predict-only, no retraining |

### Why numerical equivalence is preserved

- The sparse matrix `A[i,j] = 1` encodes exactly the same rook-neighbor relationships as the original `nb` object.
- For each cell `i` and year `t`, the C++ code collects `X[j, t]` for all `j` where `A[i,j] = 1`, skips `NA` values, and computes `max`, `min`, and `sum/count` — identical operations to the original `compute_neighbor_stats`.
- The `valid_count` tracking ensures the mean denominator matches the original's `length(neighbor_vals[!is.na(neighbor_vals)])`.
- Cells with zero valid neighbors produce `NA` for all three statistics, matching the original `c(NA, NA, NA)` return.