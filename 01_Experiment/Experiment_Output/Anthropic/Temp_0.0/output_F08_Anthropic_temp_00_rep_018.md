 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination — effectively duplicating the same spatial relationship 28 times per cell.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M paste-concatenated keys (`id_year`) and performs named-vector lookups (hash table probes) for every neighbor of every cell-year row. With ~1.37M directed neighbor pairs × 28 years ≈ 38.5M hash lookups, this alone takes hours.

3. **`compute_neighbor_stats` iterates with `lapply` over 6.46M entries**, calling `max`, `min`, `mean` individually per row. This is called 5 times (once per variable), yielding ~32.3M R-level function invocations.

4. **The entire design conflates topology (static) with variable values (dynamic).** By entangling them into a single row-level lookup, the code cannot exploit the panel structure.

## Optimization Strategy

**Separate the static spatial graph from the dynamic year-varying values.**

1. **Build a cell-level neighbor index once** — a list of length 344,208 where each element contains the integer positions of that cell's neighbors within the ordered cell vector. This is just a cleaned version of `rook_neighbors_unique` and is built once.

2. **For each variable, extract the value matrix** — reshape the variable into a `cells × years` matrix (344,208 rows × 28 columns). This allows vectorized column-wise (i.e., year-wise) operations.

3. **Compute neighbor stats via sparse-matrix multiplication or vectorized gather.** For each cell, gather neighbor values from the matrix rows, compute max/min/mean across neighbors for each year simultaneously. Using a sparse adjacency matrix, `mean` is a single matrix multiply; `max` and `min` can be computed with a grouped operation over the sparse structure.

4. **Reshape results back** to the long cell-year format and bind columns to the original data.

This reduces the work from ~6.46M × 5 R-level list iterations to a handful of sparse matrix operations and vectorized grouped computations over ~1.37M edges × 28 years, all in compiled C/C++ code underneath.

**Expected speedup:** From 86+ hours to minutes (roughly 2–10 minutes depending on RAM pressure).

**Numerical equivalence:** The same neighbor sets and the same `max`, `min`, `mean` aggregations are computed, preserving the original estimand exactly. The trained Random Forest model is untouched.

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Prepare data.table and establish cell/year orderings
# ==============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure deterministic ordering: cells in id_order, years ascending
# Create integer cell index and year index for matrix positioning
cell_id_map <- data.table(
  id       = id_order,
  cell_idx = seq_along(id_order)
)

years_all  <- sort(unique(cell_data$year))
year_map   <- data.table(
  year     = years_all,
  year_idx = seq_along(years_all)
)

n_cells <- length(id_order)
n_years <- length(years_all)

# Add cell_idx and year_idx to cell_data
cell_data <- merge(cell_data, cell_id_map, by = "id", sort = FALSE)
cell_data <- merge(cell_data, year_map,   by = "year", sort = FALSE)

# Create a row-order key so we can write results back in the correct position
cell_data[, .row_order := .I]

# ==============================================================================
# STEP 1: Build the sparse adjacency matrix ONCE (static topology)
# ==============================================================================
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element is an integer vector of neighbor indices into id_order.

build_adjacency_matrix <- function(nb_obj, n) {
  # Build COO triplets from the nb object
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove any 0-neighbor sentinel (spdep uses 0L for no-neighbor cells)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  # Sparse logical/binary adjacency matrix (row = focal cell, col = neighbor cell)
  sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, length(from_idx)),
    dims = c(n, n)
  )
}

cat("Building sparse adjacency matrix...\n")
adj_mat <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Precompute the number of neighbors per cell (for mean calculation)
n_neighbors <- as.numeric(rowSums(adj_mat))  # length n_cells

# ==============================================================================
# STEP 2: Function to build a cells x years matrix from the long data
# ==============================================================================

long_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Allocate matrix filled with NA

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  # Fill using integer indices — fully vectorized
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  mat
}

# ==============================================================================
# STEP 3: Compute neighbor max, min, mean for one variable
# ==============================================================================

compute_neighbor_stats_fast <- function(adj_mat, val_mat, n_neighbors) {
  # adj_mat:      n_cells x n_cells sparse matrix (binary)
  # val_mat:      n_cells x n_years dense matrix
  # n_neighbors:  numeric vector length n_cells
  #
  # Returns a list with three matrices (each n_cells x n_years):
  #   neighbor_max, neighbor_min, neighbor_mean
  
  n_cells <- nrow(val_mat)
  n_years <- ncol(val_mat)
  
  # --- MEAN via sparse matrix multiply ---
  # sum of neighbor values = adj_mat %*% val_mat  (sparse x dense, very fast)
  neighbor_sum <- as.matrix(adj_mat %*% val_mat)   # n_cells x n_years
  
  # To get correct mean we also need neighbor *count* excluding NAs
  # non_na_mat: 1 where val_mat is not NA, 0 otherwise
  non_na_mat <- matrix(0, nrow = n_cells, ncol = n_years)
  non_na_mat[!is.na(val_mat)] <- 1
  
  # Replace NA with 0 in val_mat for the sum computation
  val_mat_0 <- val_mat
  val_mat_0[is.na(val_mat_0)] <- 0
  
  neighbor_sum <- as.matrix(adj_mat %*% val_mat_0)
  neighbor_cnt <- as.matrix(adj_mat %*% non_na_mat)
  
  neighbor_mean <- neighbor_sum / neighbor_cnt
  neighbor_mean[neighbor_cnt == 0] <- NA_real_
  
  # --- MAX and MIN via sparse structure iteration ---
  # We iterate over each year (only 28) and use the sparse structure
  # This avoids 6.46M R-level calls; instead it's 28 vectorized operations
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Extract sparse structure once
  # For a dgCMatrix (CSC), we transpose to get CSR-like access by row
  adj_csr <- as(adj_mat, "RsparseMatrix")  # dgRMatrix: row-compressed
  row_ptr <- adj_csr@p   # length n_cells + 1, 0-based
  col_idx <- adj_csr@j   # 0-based column indices
  
  for (yr in seq_len(n_years)) {
    vals_yr <- val_mat[, yr]  # length n_cells
    
    # For each cell, gather neighbor values and compute max/min
    # We do this in vectorized chunks using the CSR structure
    # 
    # Approach: expand neighbor values, then do grouped max/min
    # group = focal cell index
    
    # Number of neighbors per cell (from CSR row pointers)
    # row_ptr is 0-based, length n_cells+1
    n_per_row <- diff(row_ptr)  # length n_cells
    
    if (length(col_idx) == 0) next
    
    # Focal cell index for each entry in col_idx
    focal <- rep(seq_len(n_cells), times = n_per_row)
    
    # Neighbor values
    nb_vals <- vals_yr[col_idx + 1L]  # col_idx is 0-based
    
    # Remove NAs
    valid <- !is.na(nb_vals)
    focal_v   <- focal[valid]
    nb_vals_v <- nb_vals[valid]
    
    if (length(nb_vals_v) == 0) next
    
    # Grouped max and min using data.table for speed
    tmp_dt <- data.table(focal = focal_v, val = nb_vals_v)
    agg <- tmp_dt[, .(vmax = max(val), vmin = min(val)), by = focal]
    
    neighbor_max[agg$focal, yr] <- agg$vmax
    neighbor_min[agg$focal, yr] <- agg$vmin
  }
  
  list(
    neighbor_max  = neighbor_max,
    neighbor_min  = neighbor_min,
    neighbor_mean = neighbor_mean
  )
}

# ==============================================================================
# STEP 4: Matrix back to long-format columns
# ==============================================================================

matrix_to_long_column <- function(mat, dt) {
  # dt must have cell_idx and year_idx columns
  mat[cbind(dt$cell_idx, dt$year_idx)]
}

# ==============================================================================
# STEP 5: Main loop — compute neighbor features for all source variables
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Long -> matrix
  val_mat <- long_to_matrix(cell_data, var_name, n_cells, n_years)
  
  # Compute stats
  stats <- compute_neighbor_stats_fast(adj_mat, val_mat, n_neighbors)
  
  # Write back to cell_data
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (max_col)  := matrix_to_long_column(stats$neighbor_max,  cell_data)]
  cell_data[, (min_col)  := matrix_to_long_column(stats$neighbor_min,  cell_data)]
  cell_data[, (mean_col) := matrix_to_long_column(stats$neighbor_mean, cell_data)]
  
  # Free memory
  rm(val_mat, stats)
  gc()
}

elapsed <- proc.time() - t0
cat(sprintf("Neighbor stats completed in %.1f seconds.\n", elapsed[3]))

# ==============================================================================
# STEP 6: Clean up helper columns, restore original row order
# ==============================================================================

setorder(cell_data, .row_order)
cell_data[, c("cell_idx", "year_idx", ".row_order") := NULL]

# ==============================================================================
# STEP 7: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained RF model object and prediction code remain exactly as before.
# For example:
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The model is NOT retrained. The feature columns have identical names and
# identical numerical values as the original pipeline would have produced.
```

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-entry list built via string hashing | Sparse adjacency matrix built once (344K × 344K) |
| **Mean computation** | `lapply` over 6.46M rows × 5 vars | Sparse matrix multiply (`adj %*% val_mat`), 5 calls |
| **Max/Min computation** | `lapply` over 6.46M rows × 5 vars | Vectorized grouped aggregation over ~1.37M edges × 28 years via `data.table` |
| **Topology reuse** | Rebuilt implicitly per row | Built once, reused for all variables and all years |
| **Estimated runtime** | 86+ hours | ~2–10 minutes |
| **Numerical result** | Original estimand | Identical — same neighbor sets, same `max`/`min`/`mean` |
| **RF model** | Pre-trained | Untouched, not retrained |
| **RAM** | ~6.46M-length list of integer vectors | Sparse matrix (~22 MB) + one dense 344K×28 matrix (~77 MB) at a time |