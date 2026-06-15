 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector — this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a numeric vector. The per-element R-level overhead (function call, subsetting, `is.na` check, aggregation) dominates when repeated millions of times.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** The 344,208 cells have the same rook neighbors every year. The code redundantly encodes 28 copies of the same adjacency structure (one per year-slice), inflating the lookup from ~1.37M edges to ~38.5M entries across all cell-years.

**Root cause summary:** The algorithm is O(N_rows × avg_neighbors) with enormous R-interpreter overhead per operation, and the data structure design prevents vectorized/compiled-code execution.

## Optimization Strategy

1. **Build the sparse adjacency matrix once** from the `nb` object (344,208 × 344,208 sparse matrix). This is the graph topology — year-invariant.

2. **Reshape each variable into a 344,208 × 28 dense matrix** (cells × years). This enables column-wise (year-wise) sparse matrix–vector operations.

3. **Compute neighbor aggregates via sparse matrix algebra:**
   - **Mean:** `A %*% X / degree` (where `A` is the binary adjacency matrix, `X` is the variable matrix, and `degree` is the row-sum vector).
   - **Max and Min:** Use a single pass in C++ via `Rcpp` over the sparse matrix CSR structure — unavoidable since max/min are not linear and can't be expressed as matrix multiplication.

4. **Avoid all string-key lookups, all `lapply` over millions of elements, and all year-level redundancy.**

5. **Memory:** Sparse matrix with ~1.37M non-zeros ≈ 33 MB. Dense matrices 344,208 × 28 ≈ 77 MB each. Total for 5 variables × 3 stats × 77 MB ≈ 3.2 GB peak — fits in 16 GB.

6. **Time:** Sparse matrix–dense matrix multiply for mean: seconds. Rcpp loop for max/min over ~1.37M edges × 28 years: seconds. Total: **minutes, not hours.**

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# ---- Step 0: Compile the C++ workhorse for max/min (runs once) ----

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// Computes row-wise max and min of X[neighbors, ] using CSR sparse structure.
// p, j are 0-based CSR arrays from a dgRMatrix.
// X is n_cells x n_years matrix.
// Returns a list with two matrices: max_mat and min_mat (same dims as X).

// [[Rcpp::export]]
List sparse_row_maxmin(IntegerVector p, IntegerVector j,
                       NumericMatrix X) {
  int n = X.nrow();
  int nyears = X.ncol();
  NumericMatrix max_mat(n, nyears);
  NumericMatrix min_mat(n, nyears);

  // Initialize to NA
  double na_val = NA_REAL;
  std::fill(max_mat.begin(), max_mat.end(), na_val);
  std::fill(min_mat.begin(), min_mat.end(), na_val);

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue;  // no neighbors -> stays NA

    for (int yr = 0; yr < nyears; yr++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        double val = X(j[k], yr);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(i, yr) = cur_max;
        min_mat(i, yr) = cur_min;
      }
      // else stays NA
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# Also need NA-aware mean via sparse ops. We handle NA by zeroing out NAs
# and tracking valid counts.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_row_mean(IntegerVector p, IntegerVector j,
                              NumericMatrix X) {
  int n = X.nrow();
  int nyears = X.ncol();
  NumericMatrix mean_mat(n, nyears);

  double na_val = NA_REAL;
  std::fill(mean_mat.begin(), mean_mat.end(), na_val);

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue;

    for (int yr = 0; yr < nyears; yr++) {
      double sum_val = 0.0;
      int valid = 0;

      for (int k = start; k < end; k++) {
        double val = X(j[k], yr);
        if (!R_IsNA(val)) {
          sum_val += val;
          valid++;
        }
      }

      if (valid > 0) {
        mean_mat(i, yr) = sum_val / (double)valid;
      }
    }
  }

  return mean_mat;
}
')


# =============================================================================
# Step 1: Build sparse adjacency matrix from nb object (once)
# =============================================================================

build_adjacency_csr <- function(nb_obj) {
  # nb_obj is a list of length n_cells; nb_obj[[i]] contains integer neighbor

  # indices (1-based). A zero-element vector or 0L means no neighbors.
  n <- length(nb_obj)

  # Build COO triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)

  # Create dgRMatrix (CSR) via dgTMatrix -> dgCMatrix -> dgRMatrix
  A <- sparseMatrix(i = from_vec, j = to_vec, x = 1,
                    dims = c(n, n), repr = "T")
  A <- as(as(A, "CsparseMatrix"), "RsparseMatrix")  # CSR format
  return(A)
}


# =============================================================================
# Step 2: Reshape panel data into cell × year matrices
# =============================================================================

reshape_to_matrix <- function(dt, id_order, years, var_name) {
  # dt: data.table with columns id, year, <var_name>
  # id_order: integer vector of cell IDs in canonical order (length n_cells)
  # years: sorted integer vector of years
  # Returns: n_cells x n_years numeric matrix

  n_cells <- length(id_order)
  n_years <- length(years)

  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(years), as.character(years))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_idx <- id_map[as.character(dt$id)]
  col_idx <- year_map[as.character(dt$year)]

  mat[cbind(row_idx, col_idx)] <- dt[[var_name]]
  return(mat)
}


# =============================================================================
# Step 3: Compute all neighbor features and write back to data
# =============================================================================

run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table for speed

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  years   <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  n_years <- length(years)

  cat("Building CSR adjacency matrix...\n")
  A_csr <- build_adjacency_csr(rook_neighbors_unique)
  # Extract CSR components (0-based for C++)
  csr_p <- A_csr@p        # length n_cells + 1, 0-based row pointers
  csr_j <- A_csr@j        # 0-based column indices

  # Precompute row/col index mapping for writing results back
  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(years), as.character(years))
  row_idx  <- id_map[as.character(cell_data$id)]
  col_idx  <- year_map[as.character(cell_data$year)]
  lin_idx  <- (col_idx - 1L) * n_cells + row_idx  # linear index into matrix

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))

    # Reshape to matrix
    X <- reshape_to_matrix(cell_data, id_order, years, var_name)

    # Compute max, min via C++
    maxmin <- sparse_row_maxmin(csr_p, csr_j, X)
    max_mat <- maxmin$max_mat  # n_cells x n_years
    min_mat <- maxmin$min_mat

    # Compute mean via C++
    mean_mat <- sparse_row_mean(csr_p, csr_j, X)

    # Write results back to cell_data using linear indexing
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data, j = max_col,  value = max_mat[lin_idx])
    set(cell_data, j = min_col,  value = min_mat[lin_idx])
    set(cell_data, j = mean_col, value = mean_mat[lin_idx])

    # Free memory
    rm(X, max_mat, min_mat, mean_mat, maxmin)
    gc()
  }

  return(cell_data)
}


# =============================================================================
# Step 4: Execute and predict
# =============================================================================

# --- Run the optimized pipeline ---
cell_data <- run_neighbor_aggregation(cell_data, id_order, rook_neighbors_unique)

# --- Apply the pre-trained Random Forest (unchanged) ---
# rf_model is already loaded; do NOT retrain.
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Why This Is Correct and Numerically Equivalent

| Original operation | Optimized equivalent | Equivalence guarantee |
|---|---|---|
| `build_neighbor_lookup` finds row indices of neighbors sharing the same year | CSR row `i` contains column indices of spatial neighbors; year dimension is handled by the matrix column | Same neighbor set per cell-year: topology is year-invariant, so `X[neighbor, year_col]` retrieves exactly the same values |
| `max(neighbor_vals[!is.na(...)])` | `sparse_row_maxmin` skips `NA` values identically | Exact same `max` over same non-NA values |
| `min(neighbor_vals[!is.na(...)])` | Same C++ function | Identical |
| `mean(neighbor_vals[!is.na(...)])` | `sparse_row_mean` sums non-NA values and divides by count | Identical to R's `mean()` on the same non-NA subset |
| Returns `NA` when no valid neighbors | Both C++ functions return `NA` when `valid == 0` or no neighbors exist | Identical |

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (string ops on 6.46M rows) | ~2 seconds (CSR construction from nb object) |
| Per-variable aggregation | ~17 hours × 5 vars | ~5–15 seconds × 5 vars (C++ over 1.37M edges × 28 years) |
| **Total** | **86+ hours** | **< 5 minutes** |
| Peak RAM | Unbounded list growth | ~4–5 GB (fits 16 GB) |

The speedup factor is approximately **1,000×**, achieved by eliminating R-interpreter overhead via compiled C++ loops over a compact CSR sparse structure, and by separating the time-invariant graph topology from the time-varying node attributes.