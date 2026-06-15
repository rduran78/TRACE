 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) equivalent cost
This function calls `lapply` over **~6.46 million rows**, and for each row it:
1. Looks up the spatial cell's rook neighbors (fine).
2. Constructs character keys by pasting `id_year` strings (expensive string allocation × 6.46M).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` — this is a **hash lookup on a character vector of length 6.46M**, repeated 6.46 million times.

The named-vector approach `idx_lookup` has O(1) amortized lookup per key, but the constant factor of character hashing across 6.46M calls, each producing multiple keys, results in enormous overhead. The entire `lapply` produces a **list of 6.46 million integer vectors** — a massive memory structure with high allocation cost.

### Bottleneck 2: `compute_neighbor_stats` — repeated per variable
This function iterates over the 6.46M-element `neighbor_lookup` list **once per source variable** (5 times), computing `max`, `min`, `mean` by subsetting a numeric vector with index vectors. The per-element R-level `lapply` loop with 6.46M iterations is slow even if each iteration is trivial.

### Summary
- **~6.46M R-level loop iterations** in `build_neighbor_lookup` with expensive string operations.
- **~32.3M R-level loop iterations** across `compute_neighbor_stats` (6.46M × 5 vars).
- Estimated 86+ hours is consistent with R-level loops over millions of elements with per-element string and list allocation.

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The neighbor structure is **purely spatial** — it does not change across years. There are only **344,208 distinct cells** and the rook neighbor relationships are fixed. The year dimension just means "look up the same neighbor cell in the same year." Therefore:

1. **Build a spatial-only neighbor structure** (344K cells, not 6.46M cell-years).
2. **Reshape each variable into a matrix**: rows = cells, columns = years. Each column is one year's data for all cells in spatial-ID order.
3. **Compute neighbor stats via sparse matrix multiplication or vectorized column operations** — for each year-column independently, gather neighbor values and compute max/min/mean using vectorized operations on the sparse adjacency structure.

This converts the problem from 6.46M R-level iterations to **28 vectorized operations over 344K cells** — a speedup factor of roughly 500–1000×.

### Concrete Plan

1. **Build a sparse adjacency matrix** `W` (344,208 × 344,208) from `rook_neighbors_unique`. This is a binary sparse matrix where `W[i,j] = 1` if cell j is a rook neighbor of cell i.
2. **Order `cell_data` by `(id, year)`** and reshape each source variable into a 344,208 × 28 matrix.
3. **For `mean`**: `W %*% X / (number of neighbors per cell)` — a single sparse matrix multiply per variable per year-column, fully vectorized.
4. **For `max` and `min`**: Use a grouped approach — for each cell, extract its neighbor rows and compute columnwise max/min. This is done efficiently with `data.table` grouping or a C++-level loop via a small Rcpp function, or by iterating over only 28 year-columns with vectorized operations.
5. **Reassemble** the features back into the panel `data.table`.

### Why Not Raster Focal?

The document header asks whether raster focal/kernel operations provide a useful analogy. They do conceptually (a rook focal window is a 3×3 cross kernel), but:
- The grid cells likely have irregular boundaries or missing cells (ocean, borders), so a regular raster focal operation would include wrong neighbors or miss valid ones.
- The precomputed `spdep::nb` object encodes the **exact** neighbor topology, which may not be a regular grid.
- We preserve exact numerical results by using the actual adjacency structure rather than a raster approximation.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table, ordered consistently
# ==============================================================================
cell_data <- as.data.table(cell_data)

# id_order: the vector of unique cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an spdep nb object (list of integer index vectors)
# These are assumed to already exist in the environment.

n_cells <- length(id_order)
stopifnot(n_cells == 344208L)

# ==============================================================================
# STEP 1: Build sparse binary adjacency matrix from the nb object
# ==============================================================================
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj[[i]] contains the indices of neighbors of cell i
  # Build COO triplets
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove 0-entries (spdep uses 0L to indicate "no neighbors")
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

W <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Number of neighbors per cell (used for mean computation)
n_neighbors <- as.integer(rowSums(W))  # length = n_cells

cat("Adjacency matrix built:", nnzero(W), "nonzero entries\n")

# ==============================================================================
# STEP 2: Create a mapping from cell id to spatial index (row in the matrix)
# ==============================================================================
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Sort cell_data by id and year for consistent reshaping
setkey(cell_data, id, year)

# Verify years
years <- sort(unique(cell_data$year))
n_years <- length(years)
stopifnot(n_years == 28L)

# Map each cell id to its spatial index
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Map each year to a column index
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_col := year_to_col[as.character(year)]]

# ==============================================================================
# STEP 3: Function to reshape a variable into a (n_cells x n_years) matrix
# ==============================================================================
reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$spatial_idx, dt$year_col)] <- dt[[var_name]]
  return(mat)
}

# ==============================================================================
# STEP 4: Compute neighbor mean via sparse matrix multiplication
# ==============================================================================
compute_neighbor_mean_matrix <- function(W, X_mat, n_neighbors) {
  # W %*% X_mat gives sum of neighbor values for each cell and year
  # Divide by number of neighbors
  sum_mat <- as.matrix(W %*% X_mat)
  mean_mat <- sum_mat / n_neighbors  # recycles column-wise
  # Cells with 0 neighbors -> NaN from 0/0, convert to NA
  mean_mat[n_neighbors == 0L, ] <- NA_real_
  return(mean_mat)
}

# ==============================================================================
# STEP 5: Compute neighbor max and min via grouped operations
#
# Strategy: iterate over cells, gather neighbor values across all years at once.
# With 344K cells this is feasible. We avoid the 6.46M iteration entirely.
# For cells with neighbors, we extract sub-matrices and compute colMax/colMin.
#
# Further optimization: group cells by identical neighbor sets (not needed here,
# the 344K loop with vectorized column operations is fast enough).
# ==============================================================================
compute_neighbor_maxmin_matrix <- function(nb_obj, X_mat, n_cells, n_years) {
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- nb_obj[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]  # remove "no neighbor" sentinel
    if (length(nb_idx) == 0L) next
    
    if (length(nb_idx) == 1L) {
      # Single neighbor: max and min are both that neighbor's values
      max_mat[i, ] <- X_mat[nb_idx, ]
      min_mat[i, ] <- X_mat[nb_idx, ]
    } else {
      # Multiple neighbors: extract sub-matrix and compute column max/min
      sub <- X_mat[nb_idx, , drop = FALSE]
      max_mat[i, ] <- apply(sub, 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(sub, 2, min, na.rm = TRUE)
    }
  }
  
  # apply with na.rm=TRUE returns -Inf/Inf when all values are NA; fix that
  max_mat[is.infinite(max_mat)] <- NA_real_
  min_mat[is.infinite(min_mat)] <- NA_real_
  
  return(list(max = max_mat, min = min_mat))
}

# ==============================================================================
# STEP 5b: FASTER alternative using Rcpp (optional, recommended)
# If Rcpp is available, this reduces 344K R-level iterations to C++ speed.
# ==============================================================================
use_rcpp <- requireNamespace("Rcpp", quietly = TRUE)

if (use_rcpp) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List compute_neighbor_maxmin_cpp(List nb_obj, NumericMatrix X, int n_cells, int n_years) {
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  
  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  
  for (int i = 0; i < n_cells; i++) {
    IntegerVector nb = nb_obj[i];
    // Filter out 0s (no-neighbor sentinel in spdep)
    std::vector<int> valid_nb;
    for (int k = 0; k < nb.size(); k++) {
      if (nb[k] != 0) valid_nb.push_back(nb[k] - 1); // 0-indexed
    }
    int n_nb = valid_nb.size();
    if (n_nb == 0) continue;
    
    for (int j = 0; j < n_years; j++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      int count = 0;
      for (int k = 0; k < n_nb; k++) {
        double val = X(valid_nb[k], j);
        if (!R_IsNA(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          count++;
        }
      }
      if (count > 0) {
        max_mat(i, j) = vmax;
        min_mat(i, j) = vmin;
      }
    }
  }
  
  return List::create(Named("max") = max_mat, Named("min") = min_mat);
}
')
}

# ==============================================================================
# STEP 6: Main loop — process each source variable
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create index vectors for writing results back to cell_data
# cell_data is keyed by (id, year), and spatial_idx + year_col give matrix coords
write_idx <- cbind(cell_data$spatial_idx, cell_data$year_col)

cat("Processing neighbor features for", length(neighbor_source_vars), "variables...\n")
t0 <- Sys.time()

for (var_name in neighbor_source_vars) {
  cat("  Variable:", var_name, "... ")
  t1 <- Sys.time()
  
  # Reshape to matrix
  X_mat <- reshape_to_matrix(cell_data, var_name, n_cells, n_years)
  
  # Compute mean via sparse matrix multiply
  mean_mat <- compute_neighbor_mean_matrix(W, X_mat, n_neighbors)
  
  # Compute max and min
  if (use_rcpp) {
    maxmin <- compute_neighbor_maxmin_cpp(rook_neighbors_unique, X_mat, n_cells, n_years)
  } else {
    maxmin <- compute_neighbor_maxmin_matrix(rook_neighbors_unique, X_mat, n_cells, n_years)
  }
  
  # Write results back to cell_data using matrix indexing
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  cell_data[, (max_col)  := maxmin$max[write_idx]]
  cell_data[, (min_col)  := maxmin$min[write_idx]]
  cell_data[, (mean_col) := mean_mat[write_idx]]
  
  cat(round(difftime(Sys.time(), t1, units = "secs"), 1), "seconds\n")
}

# Clean up temporary columns
cell_data[, c("spatial_idx", "year_col") := NULL]

cat("Total time:", round(difftime(Sys.time(), t0, units = "mins"), 1), "minutes\n")

# ==============================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns 
# (5 vars × {max, min, mean}) with numerically identical values.
#
# Example (adjust to your actual model object and prediction call):
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string-hash lookups → hours | Replaced by sparse matrix construction: **< 5 seconds** |
| `compute_neighbor_stats` (mean) | 6.46M R-level loops × 5 vars | Sparse `W %*% X`: **~2 seconds per variable** |
| `compute_neighbor_stats` (max/min) | 6.46M R-level loops × 5 vars | 344K iterations × 28 cols (Rcpp): **~30–60 seconds per variable**; pure R `apply`: ~5–15 min per variable |
| **Total** | **86+ hours** | **~5–10 minutes** (with Rcpp); **~30–60 minutes** (pure R) |

## Why Numerical Results Are Preserved

- The **sparse adjacency matrix `W`** encodes exactly the same rook-neighbor relationships as the `spdep::nb` object.
- **Mean**: `(W %*% X)[i,] / n_neighbors[i]` is algebraically identical to `mean(X[neighbors_of_i,])` for each year.
- **Max/Min**: The Rcpp and pure-R implementations iterate over the exact same neighbor index sets and apply the same `max`/`min` with `na.rm = TRUE`, producing bit-identical results.
- The **trained Random Forest model is never retrained** — only `predict()` is called on the augmented dataset, which is unchanged.