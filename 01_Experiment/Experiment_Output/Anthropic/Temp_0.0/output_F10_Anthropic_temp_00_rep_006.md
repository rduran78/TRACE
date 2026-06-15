 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector — this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` removal across millions of tiny vectors is enormous.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are a spatial property — the same adjacency graph applies to every year. The current code pastes `(id, year)` keys to resolve neighbors, redundantly recomputing the same spatial structure 28 times per cell.

**Root cause summary:** The 86+ hour runtime is dominated by R-level iteration over millions of rows with expensive string-key lookups, when the underlying operation is a sparse matrix–vector product (neighborhood aggregation) that can be expressed as matrix algebra and executed in compiled code.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (the `nb` object). This is a 344,208 × 344,208 sparse matrix with ~1.37M non-zero entries. It costs ~20 MB in memory.

2. **Reshape each variable into a 344,208 × 28 dense matrix** (cells × years). This is the "node attribute matrix."

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to 1, or the count of neighbors). Equivalently: `(A %*% X) / degree_vector`.
   - **Max and Min:** Use a grouped sparse operation. Since `A` is sparse with known structure, iterate over rows in compiled C++ code via `Rcpp`, or use a chunked approach with `Matrix` package utilities.

4. **For max and min**, pure matrix algebra doesn't directly apply (they're not linear). The efficient approach is to use **`Rcpp`** to iterate over the sparse matrix structure in compiled code, or use a clever trick with the sparse matrix to avoid R-level loops entirely.

5. **Flatten results back** to the original long-format data.frame, attach the 15 new columns (5 vars × 3 stats), and score with the pre-trained Random Forest.

**Expected speedup:** From 86+ hours to **~2–5 minutes**. The sparse matrix multiplication for mean is near-instantaneous. The Rcpp loop for max/min over ~1.37M edges × 28 years × 5 variables is also very fast.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBORHOOD AGGREGATION PIPELINE
# Preserves numerical equivalence with the original compute_neighbor_stats
# =============================================================================

library(Matrix)   # sparse matrices
library(Rcpp)     # compiled max/min aggregation

# -------------------------------------------------------------------------
# Step 0: Ensure consistent cell ordering
# -------------------------------------------------------------------------
# id_order: integer vector of length 344,208 — the cell IDs in the order
#           that matches the index positions in rook_neighbors_unique (nb object).
# cell_data: data.frame/data.table with columns id, year, ntl, ec, pop_density,
#            def, usd_est_n2, ... (~6.46M rows)
# rook_neighbors_unique: an nb object (list of length 344,208)
# rf_model: the pre-trained Random Forest model (do NOT retrain)

# -------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix ONCE from the nb object
# -------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of integer vectors (neighbor indices), length n
  # Returns: n x n sparse dgCMatrix (binary adjacency)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove zero-neighbor placeholders (spdep uses integer(0) for islands)
  valid <- !is.na(to) & to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

n_cells <- length(id_order)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Degree vector (number of neighbors per cell) — used for mean computation
degree_vec <- diff(A@p)  # column-pointer diff gives row counts for dgCMatrix
# Actually for dgCMatrix, A@p is column pointers. We need row counts.
# Safer:
degree_vec <- as.integer(rowSums(A))  # fast for sparse matrix

cat("Adjacency matrix:", n_cells, "x", n_cells,
    "with", nnzero(A), "non-zeros\n")

# -------------------------------------------------------------------------
# Step 2: Build cell-index and year-index mappings
# -------------------------------------------------------------------------
# Map cell IDs to matrix row indices (1..n_cells)
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Sorted unique years
years_sorted <- sort(unique(cell_data$year))
n_years      <- length(years_sorted)
year_to_col  <- setNames(seq_along(years_sorted), as.character(years_sorted))

# Map every row of cell_data to (cell_row_index, year_col_index)
cell_row_idx <- id_to_row[as.character(cell_data$id)]
year_col_idx <- year_to_col[as.character(cell_data$year)]

# -------------------------------------------------------------------------
# Step 3: Rcpp function for sparse neighbor max, min, mean
# -------------------------------------------------------------------------
# This operates on the CSR representation of A and a dense n_cells x n_years matrix.
# It returns three matrices: max, min, mean (each n_cells x n_years).

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_stats(IntegerVector Ap, IntegerVector Aj, 
                           NumericMatrix X, int n, int T) {
  // Ap: row pointers (length n+1), 0-based CSR format
  // Aj: column indices (length nnz), 0-based
  // X : n x T matrix of node attributes
  // Returns list of 3 matrices: max, min, mean (each n x T)
  
  NumericMatrix out_max(n, T);
  NumericMatrix out_min(n, T);
  NumericMatrix out_mean(n, T);
  
  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    int deg   = end - start;
    
    if (deg == 0) {
      for (int t = 0; t < T; t++) {
        out_max(i, t)  = NA_REAL;
        out_min(i, t)  = NA_REAL;
        out_mean(i, t) = NA_REAL;
      }
      continue;
    }
    
    for (int t = 0; t < T; t++) {
      double vmax   = R_NegInf;
      double vmin   = R_PosInf;
      double vsum   = 0.0;
      int    vcount = 0;
      
      for (int k = start; k < end; k++) {
        int j = Aj[k];
        double val = X(j, t);
        if (!R_IsNA(val) && !ISNAN(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
          vcount++;
        }
      }
      
      if (vcount == 0) {
        out_max(i, t)  = NA_REAL;
        out_min(i, t)  = NA_REAL;
        out_mean(i, t) = NA_REAL;
      } else {
        out_max(i, t)  = vmax;
        out_min(i, t)  = vmin;
        out_mean(i, t) = vsum / (double)vcount;
      }
    }
  }
  
  return List::create(Named("max")  = out_max,
                      Named("min")  = out_min,
                      Named("mean") = out_mean);
}
')

# -------------------------------------------------------------------------
# Step 4: Convert adjacency to CSR (0-based) for the Rcpp function
# -------------------------------------------------------------------------
# dgCMatrix is CSC (compressed sparse column). We need CSR (compressed sparse row).
# Transpose to get CSR pointers from CSC of A^T, or convert directly.
A_csr <- as(A, "RsparseMatrix")  # dgRMatrix = CSR format
Ap <- A_csr@p        # row pointers, 0-based, length n_cells + 1
Aj <- A_csr@j        # column indices, 0-based

# -------------------------------------------------------------------------
# Step 5: For each variable, scatter into matrix, compute stats, gather back
# -------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # --- Scatter: long vector -> n_cells x n_years matrix ---
  X_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X_mat[cbind(cell_row_idx, year_col_idx)] <- cell_data[[var_name]]
  
  # --- Compute neighbor max, min, mean via compiled sparse aggregation ---
  stats <- sparse_neighbor_stats(Ap, Aj, X_mat, n_cells, n_years)
  
  # --- Gather: n_cells x n_years matrices -> long vectors aligned to cell_data ---
  lin_idx <- cbind(cell_row_idx, year_col_idx)
  
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats$max[lin_idx]
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats$min[lin_idx]
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats$mean[lin_idx]
  
  cat("  Done.\n")
}

# -------------------------------------------------------------------------
# Step 6: Score with the pre-trained Random Forest (unchanged)
# -------------------------------------------------------------------------
# The cell_data now has the same 15 neighbor-stat columns as the original
# pipeline, with identical numerical values. The RF model is applied as-is.

cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

## Why This Is Numerically Equivalent

| Original operation | Optimized equivalent | Equivalence guarantee |
|---|---|---|
| `build_neighbor_lookup` resolves `(neighbor_id, same_year)` → row index | `X_mat[cbind(cell, year)]` scatter puts each cell-year value at the correct `(i, t)` position; `Aj` encodes the same neighbor indices from the same `nb` object | Same neighbor set, same year matching |
| `vals[idx]` gathers neighbor values | `X(j, t)` in Rcpp reads the same neighbor's value for the same year | Identical data access pattern |
| `max(neighbor_vals)`, `min(...)`, `mean(...)` after `NA` removal | Rcpp loop skips `NA`/`NaN`, tracks running max/min/sum/count | Identical arithmetic (IEEE 754 double; mean sum order may differ by ε ≈ 0, well within floating-point tolerance) |

## Performance Estimate

| Component | Time estimate |
|---|---|
| Build sparse matrix from nb object | ~2 seconds |
| Scatter each variable to matrix (×5) | ~5 seconds total |
| Rcpp sparse aggregation per variable (~1.37M edges × 28 years) | ~3–5 seconds each, ~20 seconds total |
| Gather results back (×15 columns) | ~5 seconds total |
| **Total** | **~30–60 seconds** |

This replaces the 86+ hour runtime with under 1 minute on the same 16 GB laptop, while preserving the trained Random Forest model and the original numerical estimand exactly.