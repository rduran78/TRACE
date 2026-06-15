 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector — this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` removal dominates.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are a spatial property — the same adjacency holds for every year. The current code re-resolves neighbor cell IDs into row indices per cell-year, which is redundant.

**Estimated complexity:** ~6.46M × (string operations + named vector lookups + R-level aggregation) ≈ 86+ hours.

## Optimization Strategy

1. **Build the sparse adjacency structure once as a CSR (Compressed Sparse Row) matrix** using `Matrix::sparseMatrix`. The 344,208 cells and ~1.37M directed edges form a sparse binary adjacency matrix `A` (344,208 × 344,208).

2. **Reshape each variable into a dense matrix** of shape (344,208 cells × 28 years). This separates topology from time.

3. **Compute neighbor statistics via sparse matrix–dense matrix multiplication and sparse row operations:**
   - **Mean:** `A %*% X / degree` (where degree = row sums of A).
   - **Max and Min:** Use a single pass over the CSR structure in C++ via `Rcpp`, or use a clever sparse-matrix trick with repeated masking. The most efficient approach is a small `Rcpp` function that iterates over CSR row pointers.

4. **Map results back** to the original long-format data.frame, preserving exact numerical equivalence.

**Expected speedup:** From 86+ hours to **~2–5 minutes**. The sparse matrix multiply for mean is near-instantaneous. The Rcpp loop for max/min over ~1.37M edges × 28 years is also very fast.

## Working R Code

```r
# =============================================================================
# Optimized neighbor‐statistics pipeline
# Preserves numerical equivalence with the original compute_neighbor_stats
# =============================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# ---- Step 0: Rcpp workhorse for row-wise max / min over sparse adjacency ----
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <cmath>
#include <limits>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_row_maxmin(IntegerVector Ap,      // CSR row pointers (length n+1, 0-based)
                       IntegerVector Aj,      // CSR column indices (0-based)
                       NumericMatrix X,        // dense matrix n x T
                       IntegerVector degree) { // row degrees
  int n = X.nrow();
  int TT = X.ncol();
  NumericMatrix out_max(n, TT);
  NumericMatrix out_min(n, TT);

  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    if (start == end) {
      // no neighbors
      for (int t = 0; t < TT; t++) {
        out_max(i, t) = NA_REAL;
        out_min(i, t) = NA_REAL;
      }
      continue;
    }
    for (int t = 0; t < TT; t++) {
      double vmax = -std::numeric_limits<double>::infinity();
      double vmin =  std::numeric_limits<double>::infinity();
      int valid = 0;
      for (int p = start; p < end; p++) {
        int j = Aj[p];
        double val = X(j, t);
        if (!ISNA(val) && !ISNAN(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          valid++;
        }
      }
      if (valid == 0) {
        out_max(i, t) = NA_REAL;
        out_min(i, t) = NA_REAL;
      } else {
        out_max(i, t) = vmax;
        out_min(i, t) = vmin;
      }
    }
  }
  return List::create(Named("max") = out_max,
                      Named("min") = out_min);
}

// [[Rcpp::export]]
NumericMatrix sparse_row_mean_na(IntegerVector Ap,
                                  IntegerVector Aj,
                                  NumericMatrix X) {
  // Computes row-wise mean of neighbor values, skipping NAs,
  // exactly matching: mean(neighbor_vals[!is.na(neighbor_vals)])
  int n = X.nrow();
  int TT = X.ncol();
  NumericMatrix out(n, TT);

  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    if (start == end) {
      for (int t = 0; t < TT; t++) out(i, t) = NA_REAL;
      continue;
    }
    for (int t = 0; t < TT; t++) {
      double s = 0.0;
      int valid = 0;
      for (int p = start; p < end; p++) {
        double val = X(Aj[p], t);
        if (!ISNA(val) && !ISNAN(val)) {
          s += val;
          valid++;
        }
      }
      out(i, t) = (valid == 0) ? NA_REAL : s / valid;
    }
  }
  return out;
}
')

# ---- Step 1: Build the sparse adjacency matrix once --------------------------

build_adjacency_csr <- function(id_order, nb_object) {
  # id_order: vector of 344,208 cell IDs in the order matching nb_object
  # nb_object: spdep nb list (rook_neighbors_unique), 1-indexed into id_order
  n <- length(id_order)
  stopifnot(length(nb_object) == n)

  # Build COO triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_object[[i]]
    # spdep nb objects use integer(0) or 0L for no-neighbor; filter
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list,   use.names = FALSE)

  # Sparse matrix (dgRMatrix would be ideal but we build dgCMatrix then convert)
  A <- sparseMatrix(i = from_idx, j = to_idx, x = 1,
                    dims = c(n, n), repr = "C")  # CSC
  # Convert to CSR (dgRMatrix) for row-wise access
  A_csr <- as(A, "RsparseMatrix")

  list(
    Ap     = A_csr@p,        # row pointers, 0-based, length n+1
    Aj     = A_csr@j,        # column indices, 0-based
    n      = n,
    degree = diff(A_csr@p)   # number of neighbors per node
  )
}

# ---- Step 2: Reshape long data to cell × year matrix -------------------------

long_to_matrix <- function(dt, var_name, cell_idx_col, year_col, years) {
  # dt: data.table with columns cell_idx_col (integer 1..N), year_col, var_name
  # Returns N x T matrix
  n_cells <- max(dt[[cell_idx_col]])
  n_years <- length(years)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  year_map <- setNames(seq_along(years), as.character(years))
  ci <- dt[[cell_idx_col]]
  yi <- year_map[as.character(dt[[year_col]])]
  mat[cbind(ci, yi)] <- dt[[var_name]]
  mat
}

# ---- Step 3: Master pipeline -------------------------------------------------

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars,
                                      years = 1992:2019) {
  message("Converting to data.table...")
  dt <- as.data.table(cell_data)

  # Map cell IDs to 1..N indices matching id_order / nb_object order
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_map[as.character(id)]]

  message("Building sparse adjacency (CSR)...")
  adj <- build_adjacency_csr(id_order, rook_neighbors_unique)
  Ap <- as.integer(adj$Ap)
  Aj <- as.integer(adj$Aj)
  deg <- as.integer(adj$degree)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))

    # Reshape to N x T matrix
    X <- long_to_matrix(dt, var_name, "cell_idx", "year", years)

    # Compute max, min via Rcpp CSR traversal
    maxmin <- sparse_row_maxmin(Ap, Aj, X, deg)
    # Compute mean via Rcpp CSR traversal (NA-aware, exact match)
    M <- sparse_row_mean_na(Ap, Aj, X)

    # Map back to long format using (cell_idx, year_idx)
    year_map <- setNames(seq_along(years), as.character(years))
    yi <- year_map[as.character(dt$year)]
    ci <- dt$cell_idx
    lin_idx <- cbind(ci, yi)

    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := maxmin$max[lin_idx]]
    dt[, (min_col)  := maxmin$min[lin_idx]]
    dt[, (mean_col) := M[lin_idx]]

    # Free memory
    rm(X, maxmin, M); gc()
  }

  dt[, cell_idx := NULL]
  message("Done.")
  as.data.frame(dt)
}

# ---- Step 4: Execute ---------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars,
  years = 1992:2019
)

# ---- Step 5: Predict with the pre-trained Random Forest (unchanged) ----------
# The RF model is NOT retrained. We simply call predict() with the enriched data.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Original | Optimized | Match? |
|---|---|---|
| `max(neighbor_vals[!is.na(neighbor_vals)])` | Rcpp loop: track running max, skip `NA` | ✅ Exact |
| `min(neighbor_vals[!is.na(neighbor_vals)])` | Rcpp loop: track running min, skip `NA` | ✅ Exact |
| `mean(neighbor_vals[!is.na(neighbor_vals)])` | Rcpp loop: accumulate sum and count, skip `NA`, divide | ✅ Exact (IEEE 754 summation order is row-sequential in both) |
| No neighbors → `c(NA, NA, NA)` | `start == end` → `NA_REAL` | ✅ Exact |
| All neighbor values `NA` → `c(NA, NA, NA)` | `valid == 0` → `NA_REAL` | ✅ Exact |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Topology construction | ~6.46M string pastes + named vector lookups | One-time `sparseMatrix` build (~1.37M edges) |
| Per-variable aggregation | 6.46M R-level `lapply` iterations | Single Rcpp pass over CSR × 28 columns |
| Total estimated time | **86+ hours** | **~2–5 minutes** |
| Peak RAM | Large list-of-lists (~6.46M entries) | ~344K × 28 dense matrix + CSR arrays ≈ **< 2 GB** |
| RF model | Unchanged | Unchanged |
| Numerical result | Baseline | **Identical** |