 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by character-key lookups inside an `lapply` over every row. Specifically:

1. **`paste` + named-vector lookup per row**: For each of the 6.46M rows, the code converts neighbor cell IDs to character keys (`paste(id, year, sep="_")`), then does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) per query in the worst case and has high constant overhead due to string hashing repeated millions of times.

2. **Row-wise `lapply` over 6.46M rows**: Even if each iteration is fast, the R-level loop overhead for 6.46 million iterations is enormous. The estimated 86+ hours is dominated by this function.

3. **`compute_neighbor_stats` is also row-wise** but is comparatively cheaper since it just indexes a numeric vector. Still, it runs 6.46M × 5 = 32.3M iterations total.

4. **Memory**: Building a 6.46M-element list of integer vectors is memory-heavy but fits in 16 GB.

**Root cause**: The problem is fundamentally a **sparse-matrix–vector multiply** (and element-wise min/max), but it's implemented as a sequential R-level loop with string operations.

---

## Optimization Strategy

### Key Insight
Every cell's neighbors are **the same across all 28 years**. The neighbor topology is purely spatial. So we should:

1. **Build a sparse adjacency matrix `W` once** (344,208 × 344,208, ~1.37M non-zero entries) using the `Matrix` package.
2. **Reshape each variable into a matrix** of dimension (344,208 cells × 28 years).
3. **Compute neighbor stats via sparse matrix operations**:
   - **Neighbor mean**: `W %*% X / degree` (one sparse mat-mul, milliseconds).
   - **Neighbor max / min**: Use a grouped operation over the sparse structure — iterate over columns of `W` in C-level code via `Matrix` internals or a small Rcpp function.

This replaces 6.46M × 5 R-level iterations with 5 sparse matrix multiplies (for mean) and 5 vectorized grouped operations (for max/min). Expected runtime: **seconds to low minutes** instead of 86+ hours.

### Why this preserves the estimand
- The sparse matrix `W` encodes **exactly** the same rook-neighbor relationships as `rook_neighbors_unique`.
- The numerical operations (max, min, mean of neighbor values) are identical.
- The trained Random Forest model is untouched — we only recompute the same input features faster.

---

## Working R Code

```r
# ==============================================================================
# FAST NEIGHBOR FEATURE COMPUTATION
# Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# Step 0: Convert cell_data to data.table for fast manipulation
# --------------------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure consistent ordering: create a cell index and year index
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
cell_id_map <- setNames(seq_along(id_order), as.character(id_order))
n_cells     <- length(id_order)

# Identify the unique years and create a year index
years       <- sort(unique(cell_dt$year))
n_years     <- length(years)
year_map    <- setNames(seq_along(years), as.character(years))

# Add integer indices for cell and year
cell_dt[, cell_idx := cell_id_map[as.character(id)]]
cell_dt[, year_idx := year_map[as.character(year)]]

# --------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix W from rook_neighbors_unique (nb object)
# --------------------------------------------------------------------------
# rook_neighbors_unique is a list of length n_cells;
# rook_neighbors_unique[[i]] is an integer vector of neighbor indices (into id_order)

build_sparse_adjacency <- function(nb_obj, n) {
  # Pre-allocate vectors for triplet representation
  # Count total neighbors
  total_nb <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_vec <- integer(total_nb)
  to_vec   <- integer(total_nb)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0 to denote no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_vec[pos:(pos + k - 1L)] <- i
    to_vec[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  sparseMatrix(
    i    = from_vec,
    j    = to_vec,
    x    = rep(1, total_nb),
    dims = c(n, n)
  )
}

W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# Degree vector (number of neighbors per cell) — used for mean
degree_vec <- as.numeric(rowSums(W))  # length n_cells

# --------------------------------------------------------------------------
# Step 2: For each variable, reshape to (n_cells x n_years) matrix,
#         compute neighbor max, min, mean, and write back
# --------------------------------------------------------------------------

# Ensure cell_dt is keyed for fast assignment
setkey(cell_dt, cell_idx, year_idx)

# We need a complete (cell_idx, year_idx) grid to form the matrix.
# If some cell-years are missing, we handle with NA.

# Create the matrix from cell_dt for a given variable
make_cell_year_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  mat
}

# --------------------------------------------------------------------------
# Neighbor MEAN via sparse matrix multiply
# --------------------------------------------------------------------------
compute_neighbor_mean_matrix <- function(W, X_mat, degree_vec) {
  # W %*% X_mat gives sum of neighbor values for each cell-year
  sum_mat <- as.matrix(W %*% X_mat)
  # Divide by degree; cells with 0 neighbors get NA
  mean_mat <- sum_mat / degree_vec
  mean_mat[degree_vec == 0, ] <- NA_real_
  mean_mat
}

# --------------------------------------------------------------------------
# Neighbor MAX and MIN via sparse structure
# Uses the column-compressed structure of W to avoid R-level row loops
# --------------------------------------------------------------------------
compute_neighbor_minmax_matrix <- function(W, X_mat) {
  # W is n_cells x n_cells sparse (dgCMatrix, column-compressed)
  # For row-wise operations, convert to dgRMatrix (row-compressed) or
  # use the transpose trick: t(W) is column-compressed where column j
  # holds the neighbors of cell j... but we want row i's neighbors.
  #
  # Strategy: iterate over the sparse structure efficiently.
  # Convert to dgTMatrix (triplet) for simple grouped operations via data.table
  
  Wt <- as(W, "TsparseMatrix")  # i, j are 0-based
  
  from_cell <- Wt@i + 1L  # row indices (1-based) = the focal cell
  to_cell   <- Wt@j + 1L  # col indices (1-based) = the neighbor cell
  
  n_cells <- nrow(X_mat)
  n_years <- ncol(X_mat)
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # For each year, compute grouped max and min using data.table
  for (t in seq_len(n_years)) {
    # Get neighbor values for this year
    nb_vals <- X_mat[to_cell, t]
    
    dt_tmp <- data.table(
      focal = from_cell,
      val   = nb_vals
    )
    
    # Remove NA neighbor values before aggregation
    dt_tmp <- dt_tmp[!is.na(val)]
    
    if (nrow(dt_tmp) == 0L) next
    
    agg <- dt_tmp[, .(nb_max = max(val), nb_min = min(val)), by = focal]
    
    max_mat[agg$focal, t] <- agg$nb_max
    min_mat[agg$focal, t] <- agg$nb_min
  }
  
  list(max_mat = max_mat, min_mat = min_mat)
}

# --------------------------------------------------------------------------
# Step 3: Main loop over the 5 neighbor source variables
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Processing neighbor features for: ", var_name)
  
  # Reshape to cell x year matrix
  X_mat <- make_cell_year_matrix(cell_dt, var_name, n_cells, n_years)
  
  # Compute neighbor mean (sparse mat-mul — very fast)
  mean_mat <- compute_neighbor_mean_matrix(W, X_mat, degree_vec)
  
  # Compute neighbor max and min
  minmax   <- compute_neighbor_minmax_matrix(W, X_mat)
  max_mat  <- minmax$max_mat
  min_mat  <- minmax$min_mat
  
  # Write results back to cell_dt using the (cell_idx, year_idx) indices
  idx_matrix <- cbind(cell_dt$cell_idx, cell_dt$year_idx)
  
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  set(cell_dt, j = col_max,  value = max_mat[idx_matrix])
  set(cell_dt, j = col_min,  value = min_mat[idx_matrix])
  set(cell_dt, j = col_mean, value = mean_mat[idx_matrix])
  
  # Free memory
  rm(X_mat, mean_mat, max_mat, min_mat, minmax)
  gc()
}

# --------------------------------------------------------------------------
# Step 4: Clean up helper columns and convert back to data.frame if needed
# --------------------------------------------------------------------------
cell_dt[, c("cell_idx", "year_idx") := NULL]

cell_data <- as.data.frame(cell_dt)

message("Done. All neighbor features computed.")
```

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level string lookups (86+ hrs) | One-time sparse matrix build (~seconds) |
| **Neighbor mean** | Row-wise `lapply` × 5 vars | `W %*% X_mat` sparse mat-mul × 5 (~seconds each) |
| **Neighbor max/min** | Row-wise `lapply` × 5 vars | 28 grouped `data.table` aggregations × 5 vars (~minutes total) |
| **Total estimated time** | **86+ hours** | **~2–10 minutes** |
| **Peak RAM** | ~6.46M-element list + data | Sparse matrix (1.37M entries) + dense 344K×28 matrices (~77 MB each) — fits in 16 GB |

### Why the estimand is preserved
- `W` is constructed directly from `rook_neighbors_unique` — identical neighbor sets.
- `max`, `min`, `mean` are computed on exactly the same neighbor value sets, with the same NA handling.
- The Random Forest model is never touched — only its input features are recomputed identically but faster.