 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via `paste`-based key lookups into a named character vector. Named vector lookups in R are hash-based but slow at this scale due to repeated string construction and matching for every single row.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements** in an `lapply` loop, extracting variable values, removing NAs, and computing `max/min/mean` in pure R. This is called 5 times (once per variable), meaning ~32.3 million R-level list iterations with per-element subsetting.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property—they don't change across years. But the lookup is built at the cell-year level, inflating a ~344K-node spatial graph into a ~6.46M-node spatiotemporal lookup, multiplying memory and compute by 28×.

**Root cause:** The implementation treats the panel as a flat table and rebuilds neighbor indexing at the cell-year grain, when the adjacency structure is year-invariant. Combined with R-level loops over millions of elements, this produces the 86+ hour runtime.

## Optimization Strategy

1. **Separate topology from time.** Build the sparse adjacency structure once over the 344,208 spatial cells. Rook neighbors don't change year-to-year.

2. **Use a sparse adjacency matrix.** Convert the `nb` object to a sparse `dgCMatrix` (from the `Matrix` package). This enables vectorized matrix–vector operations for neighborhood aggregation.

3. **Reshape variables into cell × year matrices.** For each source variable, pivot the panel into a 344,208 × 28 matrix. Then neighbor aggregation becomes sparse-matrix operations on columns (years), fully vectorized in compiled C code inside `Matrix`.

4. **Compute `mean` via sparse matrix multiplication**, `max` and `min` via custom sparse-row operations using `data.table` or direct CSC column traversal—all avoiding R-level per-row loops.

5. **Reassemble results** back into the original `cell_data` data.frame in the original row order, preserving numerical equivalence.

**Expected speedup:** From 86+ hours to minutes. The dominant cost becomes sparse matrix–vector multiplies and grouped aggregations, all in compiled code.

## Working R Code

```r
# =============================================================================
# Optimized Neighborhood Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is a data.table for fast operations -----------
# (We will convert back at the end if needed.)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
  was_df <- TRUE
} else {
  was_df <- FALSE
}

# ---- Step 1: Build spatial sparse adjacency matrix ONCE --------------------
# id_order: vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors into id_order)

build_sparse_adjacency <- function(id_order, nb_obj) {
  n <- length(id_order)
  # Build COO triplets from the nb list
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L or integer(0) for no-neighbor nodes
    if (length(nbrs) == 0 || (length(nbrs) == 1 && nbrs[1] == 0L)) next
    from_list[[i]] <- rep.int(i, length(nbrs))
    to_list[[i]]   <- nbrs
  }
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)
  # Directed adjacency: A[i,j] = 1 means j is a neighbor of i
  # So row i has columns = neighbors of cell i
  sparseMatrix(
    i = from_idx, j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(n, n),
    dimnames = list(as.character(id_order), as.character(id_order))
  )
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(id_order, rook_neighbors_unique)
n_cells <- length(id_order)

# Number of neighbors per cell (used for mean computation)
# This is the row sum of A — constant across years.
n_neighbors <- as.numeric(rowSums(A))  # length = n_cells

# ---- Step 2: Map cell IDs to matrix row indices ----------------------------
# Create a fast lookup from cell id -> row index in A / id_order
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# ---- Step 3: Determine year set and create cell-year ordering ---------------
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

cat(sprintf("Cells: %d, Years: %d, Cell-years: %d\n", n_cells, n_years, nrow(cell_data)))

# ---- Step 4: Create a row-index map from (cell_row, year_col) -> cell_data row
# This lets us scatter results back into the correct cell_data rows.

cat("Building cell-year index map...\n")
cell_data[, .row_idx := .I]
cell_data[, .cell_row := id_to_row[as.character(id)]]
cell_data[, .year_col := year_to_col[as.character(year)]]

# ---- Step 5: Function to pivot a variable into cell x year matrix -----------
pivot_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$.cell_row, dt$.year_col)] <- dt[[var_name]]
  mat
}

# ---- Step 6: Compute neighbor stats using sparse matrix operations ----------
# For MEAN: A %*% X gives row i = sum of neighbor values for cell i.
#   Divide by n_neighbors to get mean. Handle zero-neighbor cells -> NA.
#
# For MAX and MIN: We need true row-wise max/min over the sparse structure.
#   Strategy: use the explicit sparse structure of A to gather neighbor values
#   and compute grouped max/min via data.table.

# Pre-extract CSR-like structure of A for max/min computation
# A is stored as dgCMatrix (CSC). Convert to dgRMatrix (CSR) for row-wise access.
cat("Preparing sparse row structure for max/min...\n")
A_csr <- as(A, "RsparseMatrix")
# A_csr@j: 0-based column indices of nonzeros
# A_csr@p: row pointers (length n_cells + 1)
# For row i (0-based), nonzero column indices are A_csr@j[(A_csr@p[i]+1):A_csr@p[i+1]]

# Build a data.table of (row_cell, neighbor_col) pairs from CSR
# This is the edge list, built once and reused for all variables and years.
csr_p <- A_csr@p
csr_j <- A_csr@j  # 0-based

n_edges <- length(csr_j)
cat(sprintf("Total directed edges: %d\n", n_edges))

# Expand row indices from the row-pointer array
edge_row <- rep.int(seq_len(n_cells), diff(csr_p))  # 1-based row (cell) index
edge_col <- csr_j + 1L  # 1-based column (neighbor cell) index

# We'll create a data.table for grouped aggregation
# Columns: edge_row (the focal cell), edge_col (the neighbor cell)
edge_dt <- data.table(focal = edge_row, neighbor = edge_col)

# ---- Step 7: Main loop over neighbor source variables -----------------------

compute_and_add_all_neighbor_features <- function(cell_data, var_name, 
                                                   A, n_neighbors, 
                                                   edge_dt, 
                                                   n_cells, n_years, years) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Pivot to matrix: rows = cells, cols = years
  X <- pivot_to_matrix(cell_data, var_name, n_cells, n_years)
  
  # --- MEAN via sparse matrix multiply ---
  # A %*% X : each row i gets sum of neighbor values, per year (column)
  neighbor_sum <- as.matrix(A %*% X)  # n_cells x n_years dense matrix
  
  # Avoid division by zero: cells with no neighbors get NA
  has_neighbors <- n_neighbors > 0
  neighbor_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_mean[has_neighbors, ] <- neighbor_sum[has_neighbors, ] / n_neighbors[has_neighbors]
  
  # Where all neighbor values were NA, the sum will be NA automatically from matrix multiply.
  # But if SOME neighbors are NA, A %*% X treats NA as... well, Matrix propagates NA.
  # We need to handle partial NA exactly as the original code does:
  #   - Original removes NAs then computes mean of remaining.
  #   - If all are NA -> NA.
  #
  # To replicate: compute sum of non-NA neighbor values and count of non-NA neighbors.
  
  # Indicator of non-NA
  notNA <- ifelse(is.na(X), 0, 1)
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  
  neighbor_sum_nona  <- as.matrix(A %*% X_zero)       # sum of non-NA neighbor values
  neighbor_count_nona <- as.matrix(A %*% notNA)        # count of non-NA neighbor values
  
  neighbor_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  valid <- neighbor_count_nona > 0
  neighbor_mean[valid] <- neighbor_sum_nona[valid] / neighbor_count_nona[valid]
  
  # --- MAX and MIN via grouped aggregation on edge list ---
  # For each year, look up neighbor values, then group by focal cell.
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (yc in seq_len(n_years)) {
    # Get the variable values for this year (indexed by cell row)
    year_vals <- X[, yc]  # length n_cells
    
    # Look up neighbor values via the edge list
    nbr_vals <- year_vals[edge_dt$neighbor]
    
    # Grouped max/min by focal cell, ignoring NAs
    tmp <- data.table(focal = edge_dt$focal, val = nbr_vals)
    tmp <- tmp[!is.na(val)]
    
    if (nrow(tmp) > 0) {
      agg <- tmp[, .(mx = max(val), mn = min(val)), by = focal]
      neighbor_max[agg$focal, yc] <- agg$mx
      neighbor_min[agg$focal, yc] <- agg$mn
    }
  }
  
  # --- Scatter results back into cell_data ---
  # Use the .cell_row and .year_col indices to map matrix positions -> cell_data rows
  idx_mat <- cbind(cell_data$.cell_row, cell_data$.year_col)
  
  max_col <- paste0("n_max_", var_name)
  min_col <- paste0("n_min_", var_name)
  mean_col <- paste0("n_mean_", var_name)
  
  set(cell_data, j = max_col,  value = neighbor_max[idx_mat])
  set(cell_data, j = min_col,  value = neighbor_min[idx_mat])
  set(cell_data, j = mean_col, value = neighbor_mean[idx_mat])
  
  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_all_neighbor_features(
    cell_data, var_name, A, n_neighbors, edge_dt, n_cells, n_years, years
  )
}

# ---- Step 8: Clean up helper columns ---------------------------------------
cell_data[, c(".row_idx", ".cell_row", ".year_col") := NULL]

# Convert back to data.frame if it was one originally
if (was_df) {
  cell_data <- as.data.frame(cell_data)
}

cat("Neighbor feature computation complete.\n")

# ---- Step 9: Apply the pre-trained Random Forest model ----------------------
# The model is NOT retrained. We simply predict using the enriched cell_data.
# Adjust the predict call to match your model object name and setup.

# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary of Key Design Decisions

| Aspect | Original | Optimized |
|---|---|---|
| **Adjacency representation** | Named character vector lookup per cell-year (~6.46M entries) | Sparse matrix built once over 344K cells |
| **Neighbor mean** | R-level `lapply` + `mean()` per row | `A %*% X_zero / A %*% notNA` — sparse BLAS, vectorized across all years simultaneously |
| **Neighbor max/min** | R-level `lapply` + `max()`/`min()` per row | Edge-list gather + `data.table` grouped aggregation per year (28 iterations, compiled C internals) |
| **Memory** | 6.46M-element list of integer vectors | ~1.37M-nonzero sparse matrix + 344K×28 dense matrices (~77 MB each) — fits easily in 16 GB |
| **Total loop iterations** | ~32.3M R-level list iterations | 5 variables × (2 sparse matmuls + 28 grouped aggregations) |
| **Expected runtime** | 86+ hours | ~5–15 minutes |
| **Numerical equivalence** | Baseline | Preserved: NA handling replicates `na.rm`-then-aggregate logic exactly |
| **RF model** | Pre-trained | Untouched — predict only |

The critical insight is that **the rook adjacency graph is static across time**. By representing it as a sparse matrix and reshaping yearly data into cell×year matrices, neighborhood aggregation reduces to sparse linear algebra (for mean/sum/count) and a single pre-built edge list with `data.table` grouped aggregation (for max/min)—all executed in compiled code rather than R-level loops over 6.46 million elements.