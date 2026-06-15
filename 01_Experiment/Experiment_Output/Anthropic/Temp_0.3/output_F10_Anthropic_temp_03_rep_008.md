 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting and named-vector lookups. The `paste(id, year)` key construction and `idx_lookup[neighbor_keys]` named-vector subsetting is O(n) per lookup in the worst case, and doing this 6.46M times is catastrophic. This single function likely accounts for 60–70% of the 86+ hour runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. This is called 5 times (once per variable), so ~32.3 million R-level loop iterations with per-element allocation.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** The rook-neighbor adjacency is purely spatial — it doesn't change across years. Yet the current code rebuilds year-specific row indices for every single cell-year row, duplicating the same spatial topology 28 times with string operations.

**Key insight:** The spatial adjacency graph has only 344,208 nodes and ~1.37M directed edges. This is a small, sparse graph. The year dimension is orthogonal — every year has the same graph. We should build the topology once as a sparse matrix and use vectorized sparse matrix–dense matrix multiplication to compute neighborhood aggregates.

---

## Optimization Strategy

### Core Idea: Sparse Matrix Aggregation

1. **Build a sparse adjacency matrix `A`** (344,208 × 344,208) from the `nb` object once. Entry `A[i,j] = 1` if cell `j` is a rook neighbor of cell `i`.

2. **Build a row-degree vector `D`** where `D[i]` = number of neighbors of cell `i` (i.e., row sums of `A`).

3. **Reshape each variable into a matrix `V`** of dimension (344,208 cells × 28 years). This is the "node attribute matrix."

4. **Compute neighbor mean** as: `mean_matrix = (A %*% V) / D` — a single sparse matrix multiplication. This is O(nnz × 28) ≈ 38.4M multiply-adds, done in compiled C code via the `Matrix` package.

5. **Compute neighbor max and min** — these cannot be done via matrix multiplication. Instead, use the sparse structure of `A` to iterate in compiled code. We use `data.table` grouped operations or a compiled Rcpp routine over the CSR representation of `A`.

6. **Unroll back** to the long panel format and bind columns.

**Expected speedup:** From 86+ hours to **~2–10 minutes** depending on the max/min strategy.

### Why This Preserves Numerical Equivalence

- The sparse matrix `A` encodes exactly the same neighbor relationships as `rook_neighbors_unique`.
- `A %*% V` computes exactly `sum of neighbor values` per cell; dividing by degree gives the mean.
- Max and min are computed over exactly the same neighbor sets.
- No approximation, sampling, or model retraining is involved.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Sparse graph aggregation — numerically equivalent to original
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build sparse adjacency matrix from nb object (ONCE) ----

build_sparse_adjacency <- function(nb_obj, n) {

  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial cells (length of nb_obj)
  #
  # Returns: sparse dgCMatrix (n x n), A[i,j]=1 if j is neighbor of i

  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    nx <- x[x != 0L]  # spdep nb uses 0 for no-neighbor regions
    length(nx)
  }, integer(1))

  total_edges <- sum(edge_counts)

  # Pre-allocate triplet vectors
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs != 0L]
    k <- length(nbrs)
    if (k > 0L) {
      from_idx[pos:(pos + k - 1L)] <- i
      to_idx[pos:(pos + k - 1L)]   <- nbrs
      pos <- pos + k
    }
  }

  sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, total_edges),
    dims = c(n, n),
    repr = "C"   # CSC format, efficient for %*%
  )
}


# ---- Step 2: Reshape long panel to cell × year matrix ----

long_to_wide_matrix <- function(dt, var_name, cell_id_map, year_levels) {
  # dt: data.table with columns id, year, and var_name
  # cell_id_map: named integer vector mapping cell id -> row index (1..N)
  # year_levels: sorted unique years
  #
  # Returns: matrix (N_cells x N_years), with NA where missing

  n_cells <- length(cell_id_map)
  n_years <- length(year_levels)
  year_map <- setNames(seq_along(year_levels), as.character(year_levels))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_i <- cell_id_map[as.character(dt$id)]
  col_j <- year_map[as.character(dt$year)]
  mat[cbind(row_i, col_j)] <- dt[[var_name]]

  mat
}


# ---- Step 3: Compute neighbor MEAN via sparse matmul ----

compute_neighbor_mean <- function(A, V, degree_vec) {
  # A: sparse adjacency (N x N)
  # V: dense matrix (N x T), may contain NA
  # degree_vec: integer vector of neighbor counts per cell
  #
  # Returns: matrix (N x T) of neighbor means

  # Replace NA with 0 for multiplication, track valid counts
  V_clean <- V
  V_clean[is.na(V_clean)] <- 0

  # Indicator matrix: 1 where V is not NA
  V_valid <- matrix(1, nrow = nrow(V), ncol = ncol(V))
  V_valid[is.na(V)] <- 0

  # Sum of neighbor values (only non-NA)
  neighbor_sum   <- A %*% V_clean       # sparse %*% dense -> dense
  # Count of non-NA neighbors per cell-year
  neighbor_count <- A %*% V_valid

  # Mean = sum / count (where count > 0)
  neighbor_mean <- as.matrix(neighbor_sum) / as.matrix(neighbor_count)
  neighbor_mean[as.matrix(neighbor_count) == 0] <- NA_real_

  neighbor_mean
}


# ---- Step 4: Compute neighbor MAX and MIN via CSR iteration ----
#
# We iterate over the sparse structure. For ~1.37M edges × 28 years,
# this is fast even in R if vectorized per-row.
# For maximum speed, we use an Rcpp implementation.
# Fallback pure-R version provided below.

# --- Pure R version (still fast: ~1-3 min) ---

compute_neighbor_max_min <- function(A, V) {
  # A: sparse dgCMatrix (N x N)
  # V: dense matrix (N x T)
  #
  # Returns: list(max = matrix(N x T), min = matrix(N x T))

  # Convert to dgRMatrix (CSR) for efficient row iteration
  A_csr <- as(A, "RsparseMatrix")

  n <- nrow(V)
  n_years <- ncol(V)

  max_mat <- matrix(NA_real_, nrow = n, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n, ncol = n_years)

  # CSR: A_csr@p[i]+1 to A_csr@p[i+1] are the column indices for row i
  p <- A_csr@p
  j <- A_csr@j + 1L  # 0-based to 1-based

  for (i in seq_len(n)) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end >= start) {
      nbr_indices <- j[start:end]
      nbr_vals <- V[nbr_indices, , drop = FALSE]  # k x T matrix

      # Columnwise max/min ignoring NA
      max_mat[i, ] <- apply(nbr_vals, 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(nbr_vals, 2, min, na.rm = TRUE)

      # Fix: if all NA in a column, apply returns -Inf/Inf
      all_na <- apply(is.na(nbr_vals), 2, all)
      max_mat[i, all_na] <- NA_real_
      min_mat[i, all_na] <- NA_real_
    }
  }

  list(max = max_mat, min = min_mat)
}


# --- Rcpp version (recommended: ~10-30 sec) ---

if (requireNamespace("Rcpp", quietly = TRUE)) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List compute_max_min_csr(IntegerVector p, IntegerVector j,
                         NumericMatrix V) {
  int n = V.nrow();
  int T = V.ncol();
  NumericMatrix max_mat(n, T);
  NumericMatrix min_mat(n, T);

  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue;  // no neighbors

    for (int t = 0; t < T; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        int nbr = j[k];  // 0-based column index
        double val = V(nbr, t);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(i, t) = cur_max;
        min_mat(i, t) = cur_min;
      }
    }
  }

  return List::create(Named("max") = max_mat,
                      Named("min") = min_mat);
}
')
  USE_RCPP <- TRUE
} else {
  USE_RCPP <- FALSE
}


compute_neighbor_max_min_fast <- function(A, V) {
  A_csr <- as(A, "RsparseMatrix")
  if (USE_RCPP) {
    compute_max_min_csr(A_csr@p, A_csr@j, V)
  } else {
    compute_neighbor_max_min(A, V)
  }
}


# ---- Step 5: Compute neighbor MEAN via sparse matmul (NA-aware) ----
# Already defined above as compute_neighbor_mean


# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # cell_data: data.frame/data.table with columns: id, year, + variables
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object
  # neighbor_source_vars: character vector of variable names

  cat("Converting to data.table...\n")
  dt <- as.data.table(cell_data)
  setkey(dt, id, year)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n",
              n_cells, n_years, nrow(dt)))

  # --- Build sparse adjacency (once) ---
  cat("Building sparse adjacency matrix...\n")
  t0 <- proc.time()
  A <- build_sparse_adjacency(rook_neighbors_unique, n_cells)
  cat(sprintf("  Adjacency: %d nodes, %d directed edges (%.1f sec)\n",
              nrow(A), nnzero(A), (proc.time() - t0)[3]))

  # --- Cell ID to matrix row mapping ---
  cell_id_map <- setNames(seq_along(id_order), as.character(id_order))
  year_map    <- setNames(seq_along(years), as.character(years))

  # --- Pre-compute row/col indices for reassembly ---
  dt_row_i <- cell_id_map[as.character(dt$id)]
  dt_col_j <- year_map[as.character(dt$year)]
  dt_linear_idx <- cbind(dt_row_i, dt_col_j)

  # --- Process each variable ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    t1 <- proc.time()

    # Reshape to wide matrix (cells x years)
    V <- long_to_wide_matrix(dt, var_name, cell_id_map, years)

    # Compute neighbor mean
    cat("  Computing neighbor mean...\n")
    mean_mat <- compute_neighbor_mean(A, V, NULL)

    # Compute neighbor max and min
    cat("  Computing neighbor max/min...\n")
    maxmin <- compute_neighbor_max_min_fast(A, V)

    # Extract values back to long format using pre-computed indices
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    dt[, (max_col_name)  := maxmin$max[dt_linear_idx]]
    dt[, (min_col_name)  := maxmin$min[dt_linear_idx]]
    dt[, (mean_col_name) := mean_mat[dt_linear_idx]]

    elapsed <- (proc.time() - t1)[3]
    cat(sprintf("  Done in %.1f sec\n", elapsed))
  }

  cat("All neighbor features computed.\n")
  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================

# Load pre-saved objects
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data_enriched <- run_neighbor_feature_pipeline(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# Apply the pre-trained Random Forest (no retraining)
# predictions <- predict(rf_model, newdata = cell_data_enriched)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | Single sparse matrix build (344K nodes) |
| **Mean computation** | 6.46M R-level loops × 5 vars | 5 sparse matrix multiplications (compiled BLAS) |
| **Max/Min computation** | 6.46M R-level loops × 5 vars | CSR iteration in Rcpp (~38.4M ops in C++) |
| **Year handling** | Redundantly encoded in lookup keys | Orthogonal matrix column dimension |
| **Memory** | ~6.46M-element list of integer vectors | ~1.37M-entry sparse matrix + 344K×28 dense matrices (~77 MB each) |
| **Estimated runtime** | 86+ hours | **2–10 minutes** (Rcpp) / **10–30 min** (pure R fallback) |
| **Numerical equivalence** | Baseline | Identical — same neighbor sets, same `max/min/mean` |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |