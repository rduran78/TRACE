 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The named-vector `idx_lookup` with 6.46M entries makes each `[` lookup a hash-table probe, but doing it millions of times inside `lapply` with per-call `paste` is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** five times (once per variable). Each iteration extracts a subset of a numeric vector, removes NAs, and computes max/min/mean. The R-level `lapply` loop over 6.46M elements is inherently slow — this is called 5 times, totaling ~32.3 million R function calls.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are a spatial relationship that doesn't change across years. The current code embeds year into the lookup keys, effectively duplicating the same adjacency structure 28 times and doing string matching to resolve it.

**Root cause summary:** The 86+ hour runtime comes from ~6.46M R-level iterations with string operations (build), followed by 5 × 6.46M R-level iterations with subsetting (stats) — all in pure interpreted R with no vectorization or compiled-code acceleration.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook-neighbor graph is purely spatial (344,208 nodes). The panel has 28 years. We can:

1. **Build a sparse adjacency matrix once** from the `spdep::nb` object (344,208 × 344,208 sparse matrix). This is a one-time cost.
2. **Reshape each variable into a 344,208 × 28 matrix** (cells × years).
3. **Use sparse matrix–dense matrix multiplication** to compute neighbor sums and neighbor counts in one vectorized operation per variable. Then derive max/min/mean.

For **mean**: If `A` is the binary adjacency matrix and `X` is the cell×year matrix, then `A %*% X` gives neighbor sums, and `A %*% 1_matrix` (or row-sums of A) gives neighbor counts. Mean = sum / count.

For **max and min**: Sparse matrix multiplication doesn't directly give max/min. We must iterate over neighbors. However, we can do this **at the spatial level only** (344,208 cells, not 6.46M cell-years) using compiled C++ code via `Rcpp`, operating on the 344,208 × 28 matrix. This reduces the loop from 6.46M to 344K iterations, each processing 28 years vectorially.

### Plan

| Step | Method | Complexity |
|------|--------|------------|
| Build sparse adjacency matrix | `spdep::nb2listw` → `sparseMatrix` or direct construction | One-time, fast |
| Compute neighbor **mean** | Sparse matrix × dense matrix (`A %*% X / degree`) | Vectorized, seconds |
| Compute neighbor **max, min** | Rcpp loop over 344K cells, each accessing neighbor rows of the year-matrix | Compiled, ~seconds per variable |
| Reshape results back to panel | Matrix indexing | Vectorized |

This eliminates all string operations, reduces R-level iteration from millions to zero (or near-zero), and leverages BLAS/sparse algebra.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max, min, mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)    # sparse matrix operations
library(data.table) # fast data manipulation
library(Rcpp)      # compiled C++ for max/min

# ---- Step 0: Prepare data.table for fast manipulation ----
# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assume rook_neighbors_unique is an spdep::nb object (list of integer vectors)
# Assume id_order is the vector of cell IDs corresponding to indices 1..N in the nb object

dt <- as.data.table(cell_data)

N_cells <- length(id_order)   # 344,208
years   <- sort(unique(dt$year))  # 1992:2019
N_years <- length(years)      # 28

# Create mappings
cell_id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
year_to_col_idx        <- setNames(seq_along(years), as.character(years))

# Ensure dt is sorted by (id, year) for consistent matrix construction
setkey(dt, id, year)

# ---- Step 1: Build sparse binary adjacency matrix (one-time) ----
# From the nb object, construct a sparse matrix A of dimension N_cells x N_cells

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Construct COO triplets
  from_idx <- rep(seq_len(n), times = lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove any 0-length entries (islands) — they simply contribute no edges
  sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(n, n),
    giveCsparse = TRUE
  )
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, N_cells)
# Degree vector (number of neighbors per cell)
degree_vec <- as.numeric(rowSums(A))  # length N_cells

cat(sprintf("Adjacency matrix: %d x %d, %d non-zeros\n", 
            nrow(A), ncol(A), nnzero(A)))

# ---- Step 2: Function to reshape a variable into N_cells x N_years matrix ----

variable_to_matrix <- function(dt, var_name, cell_id_to_spatial_idx, year_to_col_idx, 
                                N_cells, N_years) {
  # Returns a matrix M where M[i, j] = value of var_name for spatial cell i in year j
  # NA for missing cell-year combinations
  
  row_idx <- cell_id_to_spatial_idx[as.character(dt$id)]
  col_idx <- year_to_col_idx[as.character(dt$year)]
  
  M <- matrix(NA_real_, nrow = N_cells, ncol = N_years)
  M[cbind(row_idx, col_idx)] <- dt[[var_name]]
  M
}

# ---- Step 3: Rcpp function for neighbor max and min ----
# This iterates over N_cells (344K) not N_rows (6.46M), and is compiled.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(IntegerVector Ap, IntegerVector Aj, 
                          NumericMatrix X, int n_cells, int n_years) {
  // Ap: row pointers of CSR sparse matrix (length n_cells + 1), 0-indexed
  // Aj: column indices of CSR sparse matrix, 0-indexed
  // X: n_cells x n_years matrix of variable values
  // Returns list with two matrices: max_mat and min_mat (n_cells x n_years)
  
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  
  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  
  for (int i = 0; i < n_cells; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    int n_neighbors = end - start;
    
    if (n_neighbors == 0) continue;
    
    for (int t = 0; t < n_years; t++) {
      double cur_max = NA_REAL;
      double cur_min = NA_REAL;
      bool found = false;
      
      for (int k = start; k < end; k++) {
        int j = Aj[k];  // neighbor spatial index (0-indexed)
        double val = X(j, t);
        if (!R_IsNA(val)) {
          if (!found) {
            cur_max = val;
            cur_min = val;
            found = true;
          } else {
            if (val > cur_max) cur_max = val;
            if (val < cur_min) cur_min = val;
          }
        }
      }
      
      if (found) {
        max_mat(i, t) = cur_max;
        min_mat(i, t) = cur_min;
      }
    }
  }
  
  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# ---- Step 4: Compute neighbor mean via sparse matrix multiplication ----

compute_neighbor_mean_matrix <- function(A, X, degree_vec) {
  # A %*% X gives sum of neighbor values for each cell-year
  # Divide by degree to get mean
  # Handle degree == 0 (islands) -> NA
  
  neighbor_sum <- as.matrix(A %*% X)  # N_cells x N_years dense matrix
  
  # Count non-NA neighbors per cell-year:
  # We need count of non-NA neighbor values, not just degree
  # Because some neighbors may have NA for that variable-year
  not_na_indicator <- ifelse(is.na(X), 0, 1)
  neighbor_count <- as.matrix(A %*% not_na_indicator)
  
  mean_mat <- neighbor_sum / neighbor_count
  mean_mat[neighbor_count == 0] <- NA_real_
  
  # Also fix the sum matrix: if a cell has neighbors but all are NA, 
  # A %*% X will give 0 (since NA was not handled). We need to zero out NAs in X first.
  mean_mat
}

compute_neighbor_mean_matrix_correct <- function(A, X_raw, degree_vec) {
  # Replace NA with 0 for multiplication, track counts separately
  X <- X_raw
  X[is.na(X)] <- 0
  
  neighbor_sum <- as.matrix(A %*% X)
  
  not_na <- ifelse(is.na(X_raw), 0, 1)
  neighbor_count <- as.matrix(A %*% not_na)
  
  mean_mat <- neighbor_sum / neighbor_count
  mean_mat[neighbor_count == 0] <- NA_real_
  mean_mat
}

# ---- Step 5: Extract CSR representation for Rcpp ----
# Matrix package stores dgCMatrix in CSC (compressed sparse column).
# We need CSR (compressed sparse row). Transpose to get CSR from CSC of A^T,
# or convert directly.

cat("Preparing CSR representation for Rcpp...\n")
A_csr <- as(A, "RsparseMatrix")  # dgRMatrix: CSR format
# A_csr@p: row pointers (0-indexed, length N_cells + 1)
# A_csr@j: column indices (0-indexed)

Ap <- A_csr@p   # integer vector, length N_cells + 1
Aj <- A_csr@j   # integer vector, 0-indexed column indices

# ---- Step 6: Main loop over variables ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need to map results back from matrix form to the data.table rows
# Create the row-index and col-index vectors for the dt rows
dt_row_spatial_idx <- cell_id_to_spatial_idx[as.character(dt$id)]
dt_col_year_idx    <- year_to_col_idx[as.character(dt$year)]
dt_matrix_idx      <- cbind(dt_row_spatial_idx, dt_col_year_idx)

cat(sprintf("Processing %d variables across %d cells x %d years (%d rows)...\n",
            length(neighbor_source_vars), N_cells, N_years, nrow(dt)))

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Variable: %s\n", var_name))
  
  # Step 6a: Reshape variable to matrix
  X_raw <- variable_to_matrix(dt, var_name, cell_id_to_spatial_idx, 
                               year_to_col_idx, N_cells, N_years)
  
  # Step 6b: Compute neighbor mean via sparse matmul
  cat("    Computing mean...\n")
  mean_mat <- compute_neighbor_mean_matrix_correct(A, X_raw, degree_vec)
  
  # Step 6c: Compute neighbor max and min via Rcpp
  cat("    Computing max and min...\n")
  maxmin <- neighbor_max_min_cpp(Ap, Aj, X_raw, N_cells, N_years)
  max_mat <- maxmin$max_mat
  min_mat <- maxmin$min_mat
  
  # Step 6d: Map back to data.table rows
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  dt[, (col_max)  := max_mat[dt_matrix_idx]]
  dt[, (col_min)  := min_mat[dt_matrix_idx]]
  dt[, (col_mean) := mean_mat[dt_matrix_idx]]
  
  # Free intermediate matrices
  rm(X_raw, mean_mat, max_mat, min_mat, maxmin)
  gc()
  
  cat(sprintf("    Done: added %s, %s, %s\n", col_max, col_min, col_mean))
}

# ---- Step 7: Convert back to data.frame if needed for predict() ----
cell_data <- as.data.frame(dt)

cat("Neighbor feature engineering complete.\n")
cat(sprintf("Final dataset: %d rows x %d columns\n", nrow(cell_data), ncol(cell_data)))

# ---- Step 8: Apply pre-trained Random Forest (unchanged) ----
# The RF model object is already in memory (e.g., `rf_model`)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **max** | `max(vals[neighbor_idx])` after NA removal | Rcpp iterates over same neighbors (from same `nb` object), same values, computes max identically | ✅ Exact |
| **min** | `min(vals[neighbor_idx])` after NA removal | Same Rcpp logic | ✅ Exact |
| **mean** | `mean(vals[neighbor_idx])` after NA removal | `(A %*% X_zeroed) / (A %*% not_na_indicator)` = sum of non-NA values / count of non-NA values = arithmetic mean | ✅ Exact (up to floating-point associativity, typically <1e-14 relative error) |

The adjacency matrix `A` is constructed from the identical `rook_neighbors_unique` nb object, so the neighbor sets are identical.

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Build lookup | ~6.46M string ops in R | One sparse matrix build (344K nodes) | ~100–500× |
| Mean (per var) | 6.46M R `lapply` calls | One sparse matmul (CHOLMOD/BLAS) | ~1000× |
| Max/Min (per var) | 6.46M R `lapply` calls | 344K Rcpp iterations × 28 years (vectorized inner loop) | ~200–500× |
| Total variables | 5 × above | 5 × above | — |
| **Estimated total** | **86+ hours** | **~2–10 minutes** | **~500–2500×** |

**Memory footprint:** The sparse matrix is ~1.37M non-zeros × 12 bytes ≈ 16 MB. Each dense cell×year matrix is 344,208 × 28 × 8 bytes ≈ 77 MB. Peak usage with 3 intermediate matrices per variable ≈ 250 MB + base data (~2–3 GB for 6.46M × 110 columns). Well within 16 GB.