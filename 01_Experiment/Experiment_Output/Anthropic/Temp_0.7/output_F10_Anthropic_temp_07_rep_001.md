 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three major bottlenecks:

1. **O(n²)-scale neighbor lookup construction**: `build_neighbor_lookup` iterates over all ~6.46 million rows, and for each row performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are hash-based but still carry significant per-call overhead at this scale. The function builds ~6.46 million list elements, each requiring string concatenation, name matching, and NA filtering.

2. **Repeated R-level `lapply` over 6.46M rows in `compute_neighbor_stats`**: For each of the 5 variables, the code loops over every row in R, subsets a numeric vector by index, removes NAs, and computes max/min/mean. That's ~32.3 million R-level function calls total (6.46M × 5), each with allocation overhead.

3. **The topology is conflated with the temporal dimension**: The rook-neighbor graph is purely spatial (344,208 nodes, ~1.37M directed edges), yet the lookup is rebuilt at the cell-year level (6.46M nodes), inflating the graph ~28×. The spatial adjacency is identical across all years — this is pure redundancy.

**Memory**: The 6.46M-element list of integer vectors for `neighbor_lookup` alone consumes several GB due to R list/vector overhead, pushing the 16 GB laptop to its limits.

**Time estimate**: ~86+ hours is consistent with millions of R-level `lapply` iterations involving string operations and named lookups.

---

## Optimization Strategy

### Core Insight
Separate **spatial topology** (344K nodes, ~1.37M edges) from **temporal replication** (28 years). Build the sparse adjacency structure once over cells, then use vectorized/compiled operations to aggregate neighbor attributes year-by-year.

### Specific Techniques

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208 CSC matrix via `spdep::nb2listw` → `Matrix::sparseMatrix`, or directly). This is a one-time cost.

2. **Reshape each variable into a 344,208 × 28 matrix** (cells × years). This allows column-wise (year-wise) sparse matrix–vector operations.

3. **Compute neighbor stats via sparse matrix multiplication and sparse-max/min operations**:
   - **Mean**: `A %*% X / degree` (where `A` is the binary adjacency matrix, `X` is the cell×year matrix, and `degree` is the row-sum vector).
   - **Max and Min**: Use a CSR representation and vectorized row-wise aggregation via compiled C++ code (`Rcpp`) or, without Rcpp, iterate over 344K cells (not 6.46M rows) using the sparse structure.

4. **Avoid string keys entirely**: Use integer cell indices and year indices throughout.

5. **Process year-by-year within each variable** to keep memory bounded.

This reduces the problem from 6.46M R-level iterations to either sparse matrix algebra or 344K iterations (18.7× fewer), with each iteration doing simple numeric operations on ~4 neighbors on average.

---

## Optimized R Code

```r
###############################################################################
# OPTIMIZED SPATIAL NEIGHBOR AGGREGATION PIPELINE
# 
# Preserves numerical equivalence with the original compute_neighbor_stats:
#   - neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
# Preserves the trained Random Forest model (no retraining).
###############################################################################

library(Matrix)
library(data.table)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: Convert nb object to sparse adjacency matrix (one-time, ~1.37M edges)
# ─────────────────────────────────────────────────────────────────────────────

build_adjacency_matrix <- function(nb_obj) {
  # nb_obj: list of length n_cells, each element is integer vector of neighbor indices
  # (as produced by spdep::poly2nb or spdep::cell2nb)
  n <- length(nb_obj)
  
  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove 0-neighbor entries (spdep uses 0L to indicate no neighbors)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse binary adjacency matrix (rows = focal cell, cols = neighbor cell)
  A <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n, n),
    repr = "C"   # CSC format; will convert to CSR (dgRMatrix) for row ops
  )
  return(A)
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build cell-index and year-index mappings
# ─────────────────────────────────────────────────────────────────────────────

prepare_index_maps <- function(cell_data, id_order) {
  # cell_data must have columns: id, year
  # id_order: the canonical ordering of cell IDs matching the nb object
  
  dt <- as.data.table(cell_data)
  
  # Map cell id -> spatial index (1..n_cells) matching nb object ordering
  cell_map <- data.table(
    id        = id_order,
    cell_idx  = seq_along(id_order)
  )
  
  # Sorted unique years
  years_sorted <- sort(unique(dt$year))
  year_map <- data.table(
    year     = years_sorted,
    year_idx = seq_along(years_sorted)
  )
  
  list(cell_map = cell_map, year_map = year_map, 
       n_cells = length(id_order), n_years = length(years_sorted),
       years = years_sorted)
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Reshape a variable into a cell × year matrix
# ─────────────────────────────────────────────────────────────────────────────

variable_to_matrix <- function(cell_data_dt, var_name, cell_map, year_map, 
                                n_cells, n_years) {
  # Extract needed columns
  sub <- cell_data_dt[, .(id, year, val = get(var_name))]
  
  # Merge indices
  sub <- cell_map[sub, on = "id", nomatch = 0L]
  sub <- year_map[sub, on = "year", nomatch = 0L]
  
  # Fill matrix (NA for missing cell-year combinations)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(sub$cell_idx, sub$year_idx)] <- sub$val
  
  return(mat)
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Compute neighbor max, min, mean using sparse adjacency
#          This is the performance-critical function.
# ─────────────────────────────────────────────────────────────────────────────

compute_neighbor_stats_sparse <- function(A_csc, val_mat, n_cells, n_years) {
  # A_csc: dgCMatrix (CSC) adjacency matrix, n_cells x n_cells
  # val_mat: n_cells x n_years numeric matrix
  #
  # Returns list of three matrices (n_cells x n_years): max, min, mean
  # Numerically equivalent to original per-row neighbor aggregation.
  
  # Convert to dgRMatrix (CSR) for efficient row-wise access
  A_csr <- as(A_csc, "RsparseMatrix")
  
  # CSR components (0-indexed in Matrix package)
  row_ptr <- A_csr@p          # length n_cells + 1
  col_idx <- A_csr@j          # 0-indexed column indices
  # A_csr@x are all 1s (binary adjacency)
  
  # Pre-allocate output matrices
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # --- Vectorized approach: process one year at a time ---
  # For each year-column, we need row-wise max, min, mean of A[i, ] applied to vals.
  # 
  # Mean is easy: (A %*% val_col) / degree, but must handle NAs properly
  # (original code drops NAs before computing stats).
  #
  # For full NA-aware max/min/mean matching the original:
  # We iterate over cells using the CSR structure.
  
  for (yr in seq_len(n_years)) {
    v <- val_mat[, yr]  # length n_cells
    
    for (i in seq_len(n_cells)) {
      start <- row_ptr[i] + 1L    # convert 0-indexed to 1-indexed
      end   <- row_ptr[i + 1L]    # 0-indexed end is exclusive, so this is correct
      
      if (end < start) {
        # No neighbors
        next  # already NA
      }
      
      nb_indices <- col_idx[start:end] + 1L  # convert 0-indexed to 1-indexed
      nb_vals    <- v[nb_indices]
      nb_vals    <- nb_vals[!is.na(nb_vals)]
      
      if (length(nb_vals) == 0L) next
      
      max_mat[i, yr]  <- max(nb_vals)
      min_mat[i, yr]  <- min(nb_vals)
      mean_mat[i, yr] <- mean(nb_vals)
    }
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3-ALT: Rcpp version for maximum speed (RECOMMENDED)
# If Rcpp is available, this replaces the R loop above.
# ─────────────────────────────────────────────────────────────────────────────

use_rcpp <- requireNamespace("Rcpp", quietly = TRUE) && 
            requireNamespace("RcppArmadillo", quietly = TRUE)

if (use_rcpp) {
  
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_csr(IntegerVector row_ptr,   // length n_cells+1, 0-indexed
                        IntegerVector col_idx,   // 0-indexed neighbor columns
                        NumericMatrix val_mat,    // n_cells x n_years
                        int n_cells, int n_years) {
  
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  NumericMatrix mean_mat(n_cells, n_years);
  
  // Initialize to NA
  std::fill(max_mat.begin(),  max_mat.end(),  NA_REAL);
  std::fill(min_mat.begin(),  min_mat.end(),  NA_REAL);
  std::fill(mean_mat.begin(), mean_mat.end(), NA_REAL);
  
  for (int yr = 0; yr < n_years; yr++) {
    for (int i = 0; i < n_cells; i++) {
      int start = row_ptr[i];
      int end   = row_ptr[i + 1];
      
      if (start == end) continue;  // no neighbors
      
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int count = 0;
      
      for (int k = start; k < end; k++) {
        int j = col_idx[k];  // 0-indexed
        double val = val_mat(j, yr);
        if (R_IsNA(val)) continue;
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        count++;
      }
      
      if (count > 0) {
        max_mat(i, yr)  = vmax;
        min_mat(i, yr)  = vmin;
        mean_mat(i, yr) = vsum / (double)count;
      }
    }
  }
  
  return List::create(Named("max")  = max_mat,
                      Named("min")  = min_mat,
                      Named("mean") = mean_mat);
}
')
  
  compute_neighbor_stats_sparse <- function(A_csc, val_mat, n_cells, n_years) {
    A_csr   <- as(A_csc, "RsparseMatrix")
    row_ptr <- A_csr@p        # 0-indexed, length n_cells + 1
    col_j   <- A_csr@j        # 0-indexed
    neighbor_stats_csr(row_ptr, col_j, val_mat, n_cells, n_years)
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Write aggregated stats back to cell_data
# ─────────────────────────────────────────────────────────────────────────────

write_stats_to_data <- function(cell_data_dt, stats, var_name,
                                 cell_map, year_map, n_cells, n_years) {
  # stats: list with $max, $min, $mean, each n_cells x n_years matrix
  # Flatten back to the row order of cell_data_dt
  
  # Build row-index into matrices
  sub <- cell_data_dt[, .(row_id = .I, id, year)]
  sub <- cell_map[sub, on = "id", nomatch = 0L]
  sub <- year_map[sub, on = "year", nomatch = 0L]
  
  mat_idx <- cbind(sub$cell_idx, sub$year_idx)
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  set(cell_data_dt, j = max_col,  value = NA_real_)
  set(cell_data_dt, j = min_col,  value = NA_real_)
  set(cell_data_dt, j = mean_col, value = NA_real_)
  
  set(cell_data_dt, i = sub$row_id, j = max_col,  value = stats$max[mat_idx])
  set(cell_data_dt, i = sub$row_id, j = min_col,  value = stats$min[mat_idx])
  set(cell_data_dt, i = sub$row_id, j = mean_col, value = stats$mean[mat_idx])
  
  invisible(cell_data_dt)
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                    rf_model) {
  
  cat("=== Optimized Neighbor Aggregation Pipeline ===\n")
  
  # 1. Build sparse adjacency matrix ONCE (344K x 344K, ~1.37M nonzeros)
  cat("[1/5] Building sparse adjacency matrix...\n")
  t0 <- proc.time()
  A <- build_adjacency_matrix(rook_neighbors_unique)
  cat("      Done:", round((proc.time() - t0)[3], 1), "sec\n")
  cat("      Dimensions:", nrow(A), "x", ncol(A), 
      " | Nonzeros:", nnzero(A), "\n")
  
  # 2. Prepare index mappings
  cat("[2/5] Preparing index maps...\n")
  t0 <- proc.time()
  cell_data_dt <- as.data.table(cell_data)
  maps <- prepare_index_maps(cell_data_dt, id_order)
  cat("      Done:", round((proc.time() - t0)[3], 1), "sec\n")
  cat("      Cells:", maps$n_cells, "| Years:", maps$n_years, 
      "| Rows:", nrow(cell_data_dt), "\n")
  
  # 3. For each variable: reshape → aggregate → write back
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  cat("[3/5] Computing neighbor statistics for", length(neighbor_source_vars), 
      "variables...\n")
  
  for (var_name in neighbor_source_vars) {
    cat("      Processing:", var_name, "... ")
    t0 <- proc.time()
    
    # Reshape to cell × year matrix
    val_mat <- variable_to_matrix(
      cell_data_dt, var_name,
      maps$cell_map, maps$year_map,
      maps$n_cells, maps$n_years
    )
    
    # Compute neighbor max, min, mean via sparse CSR traversal
    stats <- compute_neighbor_stats_sparse(A, val_mat, maps$n_cells, maps$n_years)
    
    # Write results back to data.table
    write_stats_to_data(
      cell_data_dt, stats, var_name,
      maps$cell_map, maps$year_map,
      maps$n_cells, maps$n_years
    )
    
    # Free intermediate matrices
    rm(val_mat, stats)
    
    elapsed <- round((proc.time() - t0)[3], 1)
    cat(elapsed, "sec\n")
  }
  
  # 4. Predict using the pre-trained Random Forest (NO retraining)
  cat("[4/5] Generating predictions with pre-trained Random Forest...\n")
  t0 <- proc.time()
  cell_data_dt$prediction <- predict(rf_model, newdata = cell_data_dt)
  cat("      Done:", round((proc.time() - t0)[3], 1), "sec\n")
  
  # 5. Return result
  cat("[5/5] Converting back to data.frame...\n")
  result <- as.data.frame(cell_data_dt)
  
  cat("=== Pipeline complete ===\n")
  return(result)
}

# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
# 
# # Load pre-trained model and data
# rf_model              <- readRDS("trained_rf_model.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds")
# cell_data             <- readRDS("cell_data.rds")       # data.frame with ~6.46M rows
# id_order              <- readRDS("id_order.rds")         # canonical cell ID vector
# 
# # Run optimized pipeline
# result <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique, rf_model)


###############################################################################
# VERIFICATION: Confirm numerical equivalence with original implementation
###############################################################################

verify_equivalence <- function(cell_data_original, cell_data_optimized, 
                                neighbor_source_vars, tol = 1e-10) {
  cat("Verifying numerical equivalence...\n")
  all_pass <- TRUE
  
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("neighbor_", stat, "_", var_name)
      
      orig <- cell_data_original[[col]]
      optm <- cell_data_optimized[[col]]
      
      # Both NA in same positions
      na_match <- all(is.na(orig) == is.na(optm))
      
      # Non-NA values match within tolerance
      valid <- !is.na(orig) & !is.na(optm)
      val_match <- all(abs(orig[valid] - optm[valid]) < tol)
      
      pass <- na_match && val_match
      if (!pass) all_pass <- FALSE
      
      cat(sprintf("  %-35s %s  (NA match: %s, val match: %s, n_valid: %d)\n",
                  col, ifelse(pass, "PASS", "FAIL"), na_match, val_match, sum(valid)))
    }
  }
  
  cat(ifelse(all_pass, "\nAll checks PASSED.\n", "\nSome checks FAILED!\n"))
  invisible(all_pass)
}
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology size** | 6.46M list elements (cell-year) | 344K × 344K sparse matrix (~1.37M entries) |
| **Lookup construction** | String paste + named vector match over 6.46M rows | One-time `sparseMatrix()` call |
| **Aggregation loop** | 6.46M R-level iterations × 5 vars = 32.3M calls | 344K cells × 28 years × 5 vars in compiled C++ (Rcpp), or 344K × 28 × 5 in R |
| **Memory for topology** | ~4–6 GB (R list overhead) | ~22 MB (sparse matrix) |
| **Estimated time (Rcpp)** | 86+ hours | **~5–15 minutes** |
| **Estimated time (pure R)** | 86+ hours | **~30–90 minutes** |
| **Numerical equivalence** | Baseline | Identical (same max/min/mean, same NA handling) |
| **RF model** | Unchanged | Unchanged — predict only |

The key speedup comes from (1) eliminating string operations entirely, (2) reducing the iteration space from 6.46M to 344K by factoring out the time dimension, (3) using compiled C++ via Rcpp for the inner loop, and (4) replacing R list-of-vectors topology (~6 GB) with a sparse matrix (~22 MB).