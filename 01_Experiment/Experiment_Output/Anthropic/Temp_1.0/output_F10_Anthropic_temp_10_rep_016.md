 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(n²)-scale `build_neighbor_lookup`**: For each of ~6.46M rows, it builds string keys (`paste(id, year)`), does named-vector lookups (`idx_lookup[neighbor_keys]`), and returns integer indices. Named vector lookup in R is hash-based but the sheer volume (6.46M × avg ~4 neighbors) with string allocation is extremely expensive. The lookup table `idx_lookup` itself has 6.46M named entries.

2. **`compute_neighbor_stats` with `lapply` over 6.46M elements**: Each call extracts neighbor values, removes NAs, and computes max/min/mean. This is repeated 5 times (once per variable), and each iteration touches ~6.46M list elements. Pure R loops over millions of list elements are slow.

3. **Neighbor lookup is year-coupled but topology is year-invariant**: The rook adjacency is purely spatial (344,208 cells), but the lookup is built over the full panel (6.46M rows), needlessly inflating the problem by 28×. The adjacency graph is the **same** for every year.

**Estimated cost breakdown**: ~6.46M list elements × 5 variables × 3 stats = ~97M scalar computations wrapped in R-level `lapply` + string-key construction for 6.46M rows with ~26M neighbor lookups. This explains the 86+ hour estimate.

## Optimization Strategy

### Key Insight: Separate Topology from Temporal Indexing

The rook neighbor graph is defined over **cells** (344,208 nodes), not cell-years. Every year has the identical adjacency structure. We should:

1. **Build a sparse adjacency matrix once** from the `spdep::nb` object (344,208 × 344,208 sparse matrix). This is a one-time O(cells + edges) operation.

2. **Reshape each variable into a cell × year matrix** (344,208 × 28). This is a one-time O(n) pivot.

3. **Use sparse matrix–dense matrix multiplication** to compute neighbor sums and neighbor counts simultaneously. For a binary adjacency matrix `A` and a variable matrix `X`:
   - `A %*% X` gives neighbor sums per cell per year
   - `A %*% (!is.na(X))` gives neighbor counts (for mean = sum/count)
   - For max and min: use row-wise sparse iteration (unavoidable but vectorizable)

4. **For max and min**: Convert to a CSR-like structure and use vectorized C-level operations via `data.table` or direct sparse row iteration. Alternatively, since the average degree is ~4 (1,373,394 edges / 344,208 cells ≈ 4), we can extract neighbor indices per cell (only 344K cells, not 6.46M cell-years) and vectorize over the small neighbor sets.

5. **Avoid all string key construction**. Work entirely with integer indices.

### Complexity Comparison

| Step | Original | Optimized |
|------|----------|-----------|
| Build lookup | O(6.46M) string ops | O(344K + 1.37M) integer sparse matrix |
| Compute stats | O(6.46M × 5) R-level lapply | O(344K × 28 × 5) vectorized sparse ops |
| Memory | 6.46M-element list of int vectors | 344K × 28 dense matrices + one sparse matrix |

**Expected speedup**: From 86+ hours to ~5–15 minutes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Sparse graph topology built once over cells; reused across years & variables.
# Numerically equivalent to the original build_neighbor_lookup +
# compute_neighbor_stats pipeline.
# =============================================================================

library(Matrix)    # sparse matrix operations
library(data.table) # fast reshaping and column operations

#' Build sparse binary adjacency matrix from spdep::nb object.
#' @param nb_obj  An spdep::nb object (list of integer neighbor index vectors).
#' @param n       Number of spatial cells.
#' @return A dgCMatrix (CSC sparse matrix) of dimension n x n.
build_adjacency_matrix <- function(nb_obj, n) {
  # Pre-allocate COO triplets
  # Each nb_obj[[i]] is an integer vector of neighbor indices (0 means none in

  # spdep convention, but rook_neighbors_unique should already be clean).
  from <- vector("list", n)
  to   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to denote "no neighbors" for island nodes
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from[[i]] <- rep.int(i, length(nbrs))
      to[[i]]   <- nbrs
    }
  }
  from <- unlist(from, use.names = FALSE)
  to   <- unlist(to, use.names = FALSE)

  sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n, n),
    giveCsparse = TRUE
  )
}

#' Pivot a single variable from long panel data.table into a cell x year matrix.
#' @param dt        data.table with columns: cell_idx (integer 1..N), year_idx
#'                  (integer 1..T), and the target variable column.
#' @param var_name  Character name of the variable.
#' @param n_cells   Number of cells.
#' @param n_years   Number of years.
#' @return A dense numeric matrix of dimension n_cells x n_years.
pivot_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  mat
}

#' Compute neighbor max for each cell-year using sparse adjacency.
#' Uses CSC -> CSR conversion for efficient row-wise neighbor access.
#' @param A     dgCMatrix adjacency matrix (n x n).
#' @param X     Dense numeric matrix (n x T), possibly with NAs.
#' @return Dense numeric matrix (n x T) of neighbor max values.
neighbor_max_sparse <- function(A, X) {
  n <- nrow(X)
  n_years <- ncol(X)
  # Convert to dgRMatrix (CSR) for efficient row-wise access
  A_csr <- as(A, "RsparseMatrix")
  result <- matrix(NA_real_, nrow = n, ncol = n_years)

  # A_csr@p is the row pointer (length n+1), A_csr@j is column indices (0-based)
  p <- A_csr@p
  j <- A_csr@j

  for (i in seq_len(n)) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end < start) next  # no neighbors
    nbr_idx <- j[start:end] + 1L  # convert to 1-based
    # Extract submatrix: neighbors x years
    nbr_vals <- X[nbr_idx, , drop = FALSE]
    # Column-wise max, ignoring NA
    if (length(nbr_idx) == 1L) {
      result[i, ] <- nbr_vals
    } else {
      # suppressWarnings for all-NA columns (returns -Inf, we fix below)
      col_max <- suppressWarnings(apply(nbr_vals, 2L, max, na.rm = TRUE))
      col_max[is.infinite(col_max)] <- NA_real_
      result[i, ] <- col_max
    }
  }
  result
}

#' Compute neighbor min for each cell-year using sparse adjacency.
neighbor_min_sparse <- function(A, X) {
  n <- nrow(X)
  n_years <- ncol(X)
  A_csr <- as(A, "RsparseMatrix")
  result <- matrix(NA_real_, nrow = n, ncol = n_years)
  p <- A_csr@p
  j <- A_csr@j

  for (i in seq_len(n)) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end < start) next
    nbr_idx <- j[start:end] + 1L
    nbr_vals <- X[nbr_idx, , drop = FALSE]
    if (length(nbr_idx) == 1L) {
      result[i, ] <- nbr_vals
    } else {
      col_min <- suppressWarnings(apply(nbr_vals, 2L, min, na.rm = TRUE))
      col_min[is.infinite(col_min)] <- NA_real_
      result[i, ] <- col_min
    }
  }
  result
}

#' Compute neighbor mean for each cell-year via sparse matrix multiplication.
#' mean = (A %*% X_nona) / (A %*% notNA_mask), where X_nona has 0 in place of NA.
neighbor_mean_sparse <- function(A, X) {
  X_nona <- X
  na_mask <- is.na(X)
  X_nona[na_mask] <- 0
  not_na <- matrix(1, nrow = nrow(X), ncol = ncol(X))
  not_na[na_mask] <- 0

  neighbor_sum   <- as.matrix(A %*% X_nona)     # n x T

  neighbor_count <- as.matrix(A %*% not_na)      # n x T
  neighbor_count[neighbor_count == 0] <- NA_real_ # avoid 0/0
  neighbor_sum / neighbor_count
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars,
                                       rf_model = NULL) {

  cat("Step 1: Converting to data.table and building index maps...\n")
  dt <- as.data.table(cell_data)

  # Build integer index maps for cells and years
  # id_order defines the cell ordering consistent with the nb object
  n_cells <- length(id_order)
  cell_map <- setNames(seq_along(id_order), as.character(id_order))

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_map <- setNames(seq_along(years), as.character(years))

  dt[, cell_idx := cell_map[as.character(id)]]
  dt[, year_idx := year_map[as.character(year)]]

  cat(sprintf("  Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))

  # ---- Step 2: Build sparse adjacency matrix (one-time) --------------------
  cat("Step 2: Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  cat(sprintf("  Adjacency matrix: %d x %d with %d non-zeros\n",
              nrow(A), ncol(A), nnz(A)))

  # Pre-convert to CSR once for max/min (shared across all variables)
  A_csr <- as(A, "RsparseMatrix")
  p_csr <- A_csr@p
  j_csr <- A_csr@j

  # ---- Step 3: For each variable, pivot -> compute stats -> unpivot ---------
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Step 3: Processing variable '%s'...\n", var_name))

    # Pivot to cell x year matrix
    X <- pivot_to_matrix(dt, var_name, n_cells, n_years)

    # --- Neighbor Mean (via sparse matmul, extremely fast) ---
    cat("  Computing neighbor mean (sparse matmul)...\n")
    mean_mat <- neighbor_mean_sparse(A, X)

    # --- Neighbor Max and Min (row-wise sparse iteration) ---
    # Optimized: single pass over CSR for both max and min simultaneously
    cat("  Computing neighbor max and min (CSR row iteration)...\n")
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (i in seq_len(n_cells)) {
      start <- p_csr[i] + 1L
      end   <- p_csr[i + 1L]
      if (end < start) next
      nbr_idx <- j_csr[start:end] + 1L
      nbr_vals <- X[nbr_idx, , drop = FALSE]  # (degree x n_years)

      if (length(nbr_idx) == 1L) {
        max_mat[i, ] <- nbr_vals
        min_mat[i, ] <- nbr_vals
      } else {
        for (t in seq_len(n_years)) {
          v <- nbr_vals[, t]
          v <- v[!is.na(v)]
          if (length(v) > 0L) {
            max_mat[i, t] <- max(v)
            min_mat[i, t] <- min(v)
          }
        }
      }
    }

    # --- Unpivot back to long format and merge into dt ---
    cat("  Unpivoting and merging results...\n")
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Use matrix indexing for direct assignment (avoids reshape overhead)
    idx <- cbind(dt$cell_idx, dt$year_idx)
    dt[, (max_col)  := max_mat[idx]]
    dt[, (min_col)  := min_mat[idx]]
    dt[, (mean_col) := mean_mat[idx]]

    # Free memory
    rm(X, mean_mat, max_mat, min_mat)
    gc(verbose = FALSE)

    cat(sprintf("  Done: added %s, %s, %s\n", max_col, min_col, mean_col))
  }

  # ---- Step 4: Clean up temp columns and return ----------------------------
  dt[, c("cell_idx", "year_idx") := NULL]

  cat("Pipeline complete.\n")
  return(dt)
}

# =============================================================================
# FURTHER OPTIMIZATION: Rcpp-accelerated max/min pass
# The R-level loop over 344K cells with inner loops over 28 years is the
# remaining bottleneck (~minutes). This Rcpp version reduces it to seconds.
# =============================================================================

# If Rcpp is available, this replaces the R-level row iteration:

rcpp_neighbor_maxmin_code <- '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_csr(IntegerVector p, IntegerVector j,
                         NumericMatrix X) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix max_mat(n, T);
  NumericMatrix min_mat(n, T);

  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    int start = p[i];      // 0-based CSR pointers
    int end   = p[i + 1];
    if (end <= start) continue;

    for (int t = 0; t < T; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int count = 0;

      for (int k = start; k < end; k++) {
        int nbr = j[k];  // 0-based column index
        double val = X(nbr, t);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          count++;
        }
      }

      if (count > 0) {
        max_mat(i, t) = cur_max;
        min_mat(i, t) = cur_min;
      }
    }
  }

  return List::create(Named("max") = max_mat,
                      Named("min") = min_mat);
}
'

# To use the Rcpp version, call this enhanced pipeline:
optimize_neighbor_features_fast <- function(cell_data,
                                             id_order,
                                             rook_neighbors_unique,
                                             neighbor_source_vars,
                                             rf_model = NULL,
                                             use_rcpp = TRUE) {

  if (use_rcpp) {
    if (requireNamespace("Rcpp", quietly = TRUE)) {
      cat("Compiling Rcpp neighbor max/min kernel...\n")
      Rcpp::sourceCpp(code = rcpp_neighbor_maxmin_code)
    } else {
      cat("Rcpp not available; falling back to pure R.\n")
      use_rcpp <- FALSE
    }
  }

  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  cell_map <- setNames(seq_along(id_order), as.character(id_order))
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_map <- setNames(seq_along(years), as.character(years))

  dt[, cell_idx := cell_map[as.character(id)]]
  dt[, year_idx := year_map[as.character(year)]]

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))

  # Build sparse adjacency once
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  A_csr <- as(A, "RsparseMatrix")
  cat(sprintf("Adjacency: %d non-zeros, avg degree %.1f\n",
              nnz(A), nnz(A) / n_cells))

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing '%s'...\n", var_name))
    X <- pivot_to_matrix(dt, var_name, n_cells, n_years)

    # Mean via sparse matmul
    mean_mat <- neighbor_mean_sparse(A, X)

    # Max and Min
    if (use_rcpp) {
      mm <- neighbor_maxmin_csr(A_csr@p, A_csr@j, X)
      max_mat <- mm$max
      min_mat <- mm$min
    } else {
      # Pure R fallback (still much faster than original due to cell-level loop)
      max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
      min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
      p_csr <- A_csr@p
      j_csr <- A_csr@j
      for (i in seq_len(n_cells)) {
        start <- p_csr[i] + 1L
        end   <- p_csr[i + 1L]
        if (end < start) next
        nbr_idx <- j_csr[start:end] + 1L
        nbr_vals <- X[nbr_idx, , drop = FALSE]
        if (length(nbr_idx) == 1L) {
          max_mat[i, ] <- nbr_vals
          min_mat[i, ] <- nbr_vals
        } else {
          for (t in seq_len(n_years)) {
            v <- nbr_vals[, t]
            v <- v[!is.na(v)]
            if (length(v) > 0L) {
              max_mat[i, t] <- max(v)
              min_mat[i, t] <- min(v)
            }
          }
        }
      }
    }

    # Map back to long format
    idx <- cbind(dt$cell_idx, dt$year_idx)
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := max_mat[idx]]
    dt[, (min_col)  := min_mat[idx]]
    dt[, (mean_col) := mean_mat[idx]]

    rm(X, mean_mat, max_mat, min_mat)
    gc(verbose = FALSE)
    cat(sprintf("  Added %s, %s, %s\n", max_col, min_col, mean_col))
  }

  dt[, c("cell_idx", "year_idx") := NULL]
  cat("Neighbor features complete.\n")

  # ---- Apply pre-trained RF (no retraining) ----
  if (!is.null(rf_model)) {
    cat("Generating predictions with pre-trained Random Forest...\n")
    dt[, rf_prediction := predict(rf_model, newdata = dt)]
  }

  return(dt)
}

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
# # Load your data and pre-trained model:
# cell_data <- readRDS("cell_data.rds")
# id_order  <- readRDS("id_order.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds")
# rf_model  <- readRDS("rf_model.rds")
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# result <- optimize_neighbor_features_fast(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = neighbor_source_vars,
#   rf_model               = rf_model,
#   use_rcpp               = TRUE
# )
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup granularity** | 6.46M cell-years (string keys) | 344K cells (integer indices) |
| **Adjacency representation** | R list with string hashing | Sparse CSR matrix (integer arrays) |
| **Mean computation** | R `lapply` over 6.46M elements | Sparse matrix × dense matrix multiply (BLAS-accelerated) |
| **Max/Min computation** | R `lapply` over 6.46M elements | Loop over 344K cells with vectorized column access (or Rcpp single-pass) |
| **Topology reuse** | Rebuilt implicitly per row | Built once, reused across 5 variables × 28 years |
| **Memory pattern** | 6.46M-element list of integer vectors | 344K × 28 dense matrices + one sparse matrix (~150 MB total) |
| **Estimated runtime** | 86+ hours | ~2–10 minutes (Rcpp) / ~20–40 minutes (pure R) |
| **Numerical equivalence** | Baseline | Identical: same max, min, mean over same non-NA neighbor values |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |