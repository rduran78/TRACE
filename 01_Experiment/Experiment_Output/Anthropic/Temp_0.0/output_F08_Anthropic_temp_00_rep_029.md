 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years together into string keys, performs string matching via named vectors for every single row, and repeats all of this redundantly for every year a cell appears. This means:

1. **Redundant work × 28**: The neighbor graph is static — cell *i*'s neighbors are the same in 1992 as in 2019. Yet the lookup is rebuilt for every cell-year combination, inflating a 344K-element problem into a 6.46M-element problem.
2. **String-key hashing is slow**: `paste(..., sep="_")` and named-vector lookup (`idx_lookup[neighbor_keys]`) over millions of entries is extremely expensive in R.
3. **`compute_neighbor_stats` iterates 6.46M list entries**: Each call to `lapply` over the full row-level lookup, repeated for 5 variables, means ~32.3 million list-element iterations, each doing subsetting, `max`, `min`, `mean`.
4. **Memory pressure**: Storing 6.46M integer vectors in a list, plus repeated copying of `cell_data`, risks exceeding 16 GB RAM.

**In summary**: The static neighbor topology is being entangled with the dynamic year dimension, causing a 28× blowup in both time and memory.

---

## Optimization Strategy

**Separate the static topology from the dynamic variable values.**

1. **Build the neighbor lookup once, at the cell level (344K entries), not the cell-year level (6.46M entries).** The `rook_neighbors_unique` nb object already encodes this — we just need a clean integer-index mapping from cell ID to position.

2. **Compute neighbor stats using vectorized matrix operations.** Reshape each variable into a `cells × years` matrix. Then for each cell, its neighbor rows in the matrix are fixed. We can compute `max`, `min`, `mean` across neighbor rows for all 28 years simultaneously using column-wise operations on submatrices — or better yet, use a sparse-matrix approach.

3. **Use a sparse adjacency matrix and matrix multiplication** for the neighbor mean (which is a linear operation). For min and max, use an efficient row-grouped approach over the sparse structure.

4. **Avoid per-element `lapply` over millions of rows.** Instead, operate on vectors/matrices.

**Expected speedup**: From ~86 hours down to minutes. The bottleneck becomes ~344K cells × 5 variables with vectorized column operations over 28 years, plus sparse matrix multiplications.

**Numerical equivalence**: The sparse-matrix mean uses the same arithmetic (sum of neighbor values / count of neighbors). Min and max are computed from the same neighbor sets. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the static neighbor structure ONCE (cell-level, not cell-year)
# ==============================================================================

# id_order: vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_sparse_adjacency <- function(id_order, neighbors) {
  # neighbors is an nb object: list of length n_cells,

  # each element is an integer vector of neighbor indices (into id_order)
  # with 0-neighbor cells represented as 0L per spdep convention.
  
  n <- length(id_order)
  
  # Build COO (coordinate) representation
  from <- rep(seq_len(n), times = vapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  to <- unlist(lapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else x
  }), use.names = FALSE)
  
  # Sparse adjacency matrix: A[i,j] = 1 if j is a neighbor of i

  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  # Degree vector (number of neighbors per cell)
  degree <- rowSums(A)  # integer-valued numeric vector
  
  list(A = A, degree = degree)
}

# ==============================================================================
# STEP 2: Compute neighbor stats via matrix operations (all years at once)
# ==============================================================================

compute_neighbor_features_matrix <- function(cell_dt, id_order, adj) {
  # cell_dt: data.table with columns id, year, and the variable columns
  # id_order: vector of cell IDs defining row order in adjacency matrix
  # adj: list with A (sparse adjacency) and degree (neighbor counts)
  
  A      <- adj$A
  degree <- adj$degree
  n      <- length(id_order)
  
  # Create a mapping from cell id to matrix row index
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  
  # Ensure data.table is keyed for fast access
  setkey(cell_dt, id, year)
  
  # Get sorted unique years
  years <- sort(unique(cell_dt$year))
  n_years <- length(years)
  
  # Map each row of cell_dt to its matrix-row index
  cell_dt[, mat_row := id_to_row[as.character(id)]]
  
  # Create a year-to-column mapping
  year_to_col <- setNames(seq_along(years), as.character(years))
  cell_dt[, mat_col := year_to_col[as.character(year)]]
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)
    
    # ------------------------------------------------------------------
    # Build cells x years matrix for this variable
    # ------------------------------------------------------------------
    vals <- cell_dt[[var_name]]
    
    # Dense matrix: rows = cells (in id_order), cols = years
    V <- matrix(NA_real_, nrow = n, ncol = n_years)
    V[cbind(cell_dt$mat_row, cell_dt$mat_col)] <- vals
    
    # ------------------------------------------------------------------
    # NEIGHBOR MEAN via sparse matrix multiplication
    # ------------------------------------------------------------------
    # A %*% V gives, for each cell i and year t, the SUM of neighbor values
    # Divide by degree to get mean
    neighbor_sum <- as.matrix(A %*% V)  # n x n_years dense matrix
    
    # Where degree == 0, result should be NA
    neighbor_mean <- neighbor_sum / degree  # recycles degree along columns
    neighbor_mean[degree == 0, ] <- NA_real_
    
    # Also set to NA where ALL neighbors had NA values
    # Count non-NA neighbors: replace NA with 0 in V, multiply by A
    V_notna <- ifelse(is.na(V), 0, 1)
    V_zeroed <- V
    V_zeroed[is.na(V_zeroed)] <- 0
    
    neighbor_count_notna <- as.matrix(A %*% V_notna)  # count of non-NA neighbors
    neighbor_sum_clean   <- as.matrix(A %*% V_zeroed)  # sum excluding NAs
    
    neighbor_mean_clean <- neighbor_sum_clean / neighbor_count_notna
    neighbor_mean_clean[neighbor_count_notna == 0] <- NA_real_
    
    # ------------------------------------------------------------------
    # NEIGHBOR MAX and MIN: iterate over cells (but only 344K, not 6.46M)
    # Use the sparse matrix structure to extract neighbor indices once
    # ------------------------------------------------------------------
    neighbor_max <- matrix(NA_real_, nrow = n, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n, ncol = n_years)
    
    # Extract neighbor indices from sparse matrix (CSC format)
    # Convert to row-oriented list for efficient access
    # Using the nb object directly is actually most efficient here
    # But we can also extract from the sparse matrix
    
    # We'll use a chunked approach for memory efficiency
    # Process in chunks of cells
    chunk_size <- 10000L
    n_chunks <- ceiling(n / chunk_size)
    
    for (ch in seq_len(n_chunks)) {
      start_i <- (ch - 1L) * chunk_size + 1L
      end_i   <- min(ch * chunk_size, n)
      
      for (i in start_i:end_i) {
        nb_idx <- adj$nb_indices[[i]]
        if (length(nb_idx) == 0L) next
        
        # Extract the submatrix of neighbor values: |neighbors| x n_years
        nb_vals <- V[nb_idx, , drop = FALSE]  # matrix
        
        # Column-wise max and min, handling NAs
        if (length(nb_idx) == 1L) {
          neighbor_max[i, ] <- nb_vals[1, ]
          neighbor_min[i, ] <- nb_vals[1, ]
        } else {
          # suppressWarnings to handle all-NA columns
          neighbor_max[i, ] <- suppressWarnings(apply(nb_vals, 2, max, na.rm = TRUE))
          neighbor_min[i, ] <- suppressWarnings(apply(nb_vals, 2, min, na.rm = TRUE))
        }
      }
    }
    
    # Fix Inf/-Inf from all-NA columns
    neighbor_max[is.infinite(neighbor_max)] <- NA_real_
    neighbor_min[is.infinite(neighbor_min)] <- NA_real_
    
    # ------------------------------------------------------------------
    # Write results back to cell_dt
    # ------------------------------------------------------------------
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_dt[, (max_col)  := neighbor_max[cbind(mat_row, mat_col)]]
    cell_dt[, (min_col)  := neighbor_min[cbind(mat_row, mat_col)]]
    cell_dt[, (mean_col) := neighbor_mean_clean[cbind(mat_row, mat_col)]]
  }
  
  # Clean up helper columns
  cell_dt[, c("mat_row", "mat_col") := NULL]
  
  return(cell_dt)
}

# ==============================================================================
# STEP 3: Optimized min/max using nb list directly (avoids sparse extraction)
# ==============================================================================

# More efficient version that stores nb indices as a simple list of integer vectors

prepare_nb_indices <- function(neighbors) {
  # Convert spdep nb object to clean list of integer vectors
  lapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else as.integer(x)
  })
}

# ==============================================================================
# FULL PIPELINE
# ==============================================================================

run_optimized_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table if not already
  cell_dt <- as.data.table(cell_data)
  
  message("Building static adjacency structure (once)...")
  
  # --- Static structure: built once for all years ---
  adj <- build_sparse_adjacency(id_order, rook_neighbors_unique)
  adj$nb_indices <- prepare_nb_indices(rook_neighbors_unique)
  
  message("Computing neighbor features for all variables and years...")
  
  cell_dt <- compute_neighbor_features_matrix(cell_dt, id_order, adj)
  
  message("Done. Neighbor features added.")
  
  return(cell_dt)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

# cell_data <- run_optimized_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with the pre-trained Random Forest prediction as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Vectorized Min/Max Without `apply`

The `apply(..., 2, max)` inside the cell loop can be slow for cells with many neighbors. Here is a faster alternative using `matrixStats` or Rcpp for the min/max step:

```r
# ==============================================================================
# OPTIONAL: Rcpp accelerated neighbor min/max (replaces the R loop)
# ==============================================================================

Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_minmax_cpp(NumericMatrix V, List nb_indices) {
  int n = nb_indices.size();
  int T = V.ncol();
  
  NumericMatrix out_max(n, T);
  NumericMatrix out_min(n, T);
  
  // Initialize with NA
  std::fill(out_max.begin(), out_max.end(), NA_REAL);
  std::fill(out_min.begin(), out_min.end(), NA_REAL);
  
  for (int i = 0; i < n; i++) {
    IntegerVector idx = nb_indices[i];
    int k = idx.size();
    if (k == 0) continue;
    
    for (int t = 0; t < T; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      int count = 0;
      
      for (int j = 0; j < k; j++) {
        double val = V(idx[j] - 1, t);  // R is 1-indexed
        if (!R_IsNA(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          count++;
        }
      }
      
      if (count > 0) {
        out_max(i, t) = vmax;
        out_min(i, t) = vmin;
      }
    }
  }
  
  return List::create(Named("max") = out_max, Named("min") = out_min);
}
')

# Replace the R loop in compute_neighbor_features_matrix with:
# mm <- neighbor_minmax_cpp(V, adj$nb_indices)
# neighbor_max <- mm$max
# neighbor_min <- mm$min
```

---

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup granularity** | Per cell-year (6.46M entries) | Per cell (344K entries) |
| **Lookup construction** | String pasting + named vector matching | Integer index mapping + sparse matrix |
| **Neighbor mean** | `lapply` over 6.46M elements | Sparse matrix multiply (`A %*% V`), all years at once |
| **Neighbor min/max** | `lapply` over 6.46M elements | Loop over 344K cells, vectorized across 28 years (or Rcpp) |
| **Total iterations** | ~32.3M (6.46M × 5 vars) | ~1.72M (344K × 5 vars), each doing 28 years vectorized |
| **Estimated time** | 86+ hours | Minutes (pure R) to seconds (with Rcpp) |
| **RF model** | Unchanged | Unchanged |
| **Numerical result** | — | Identical (same max, min, mean over same neighbor sets) |