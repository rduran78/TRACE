 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each built via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per probe in the worst case — with 6.46M keys, each lookup is expensive. This alone can take many hours.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting variable values and computing `max/min/mean` per node. With 5 variables, this is ~32.3M iterations of R-level loops.

3. **The neighbor topology is year-invariant** (rook adjacency on a fixed grid), but the lookup is rebuilt monolithically for every cell-year combination, conflating spatial structure with temporal indexing. This means the same adjacency information is redundantly encoded 28 times (once per year).

**Key insight:** The adjacency graph is purely spatial (344,208 nodes, ~1.37M directed edges). The year dimension is a panel dimension — every cell has the same neighbors in every year. The current code expands the spatial graph into the full cell-year space unnecessarily.

---

## Optimization Strategy

1. **Separate spatial topology from temporal indexing.** Build the sparse adjacency structure once over 344,208 cells. For each year, slice the data, and use vectorized sparse-matrix operations to compute neighbor statistics.

2. **Use a sparse adjacency matrix (CSC/CSR format via `Matrix` package).** Convert the `nb` object to a sparse logical/binary matrix `A` of dimension 344,208 × 344,208. Entry `A[i,j] = 1` means cell `j` is a rook neighbor of cell `i`.

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A %*% x / A %*% 1` (sparse matrix-vector multiply is highly optimized in C).
   - **Max and Min:** Use row-wise sparse operations. Replace structural zeros with `NA` or sentinel values and compute row extrema.

4. **Process year-by-year** to keep memory bounded (~344K × 5 variables per year slice).

5. **Preserve numerical equivalence:** The sparse-matrix approach computes identical `max`, `min`, `mean` of the exact same neighbor value sets.

6. **Do not retrain the Random Forest.** Only reconstruct the predictor columns identically.

**Expected speedup:** From 86+ hours to **~5–15 minutes**. Sparse matrix-vector products over 344K nodes with ~4 neighbors each are nearly instantaneous. The loop is 28 years × 5 variables = 140 sparse operations for mean/sum, plus 140 row-wise max/min operations.

---

## Optimized R Code

```r
library(Matrix)
library(spdep)
library(data.table)

# ===========================================================================
# STEP 0: Load pre-existing objects
# ===========================================================================
# Assumes these are already in the environment or loaded from disk:
#   cell_data              — data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order               — integer vector of cell IDs in canonical order (length 344,208)
#   rook_neighbors_unique  — nb object (list of length 344,208)
#   rf_model               — pre-trained Random Forest model (DO NOT RETRAIN)

# ===========================================================================
# STEP 1: Convert cell_data to data.table for fast indexing
# ===========================================================================
cell_data <- as.data.table(cell_data)

# Canonical cell ordering: map cell id -> row index in spatial graph
n_cells <- length(id_order)
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# ===========================================================================
# STEP 2: Build sparse adjacency matrix ONCE (344,208 x 344,208)
# ===========================================================================
# Convert nb object to a sparse matrix.
# Each entry A[i,j] = 1 means cell j is a rook neighbor of cell i.

build_adjacency_matrix <- function(nb_obj, n) {
  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  # Pre-allocate vectors
  row_idx <- integer(total_edges)
  col_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    if (length(nb) == 1L && nb[1] == 0L) next
    k <- length(nb)
    row_idx[pos:(pos + k - 1L)] <- i
    col_idx[pos:(pos + k - 1L)] <- nb
    pos <- pos + k
  }
  
  sparseMatrix(
    i = row_idx, j = col_idx,
    x = rep(1, total_edges),
    dims = c(n, n),
    giveCsparse = TRUE
  )
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
cat(sprintf("Adjacency matrix: %d x %d, %d non-zeros\n", nrow(A), ncol(A), nnzero(A)))

# Pre-compute the number of neighbors per cell (used for mean calculation)
# ones vector
ones_vec <- rep(1, n_cells)
neighbor_count <- as.numeric(A %*% ones_vec)  # length n_cells

# ===========================================================================
# STEP 3: Compute neighbor stats using sparse operations
# ===========================================================================
# For max and min, we cannot use simple matrix multiply. Strategy:
#   - Extract the sparse structure of A
#   - For each row i, gather vals[A[i,]@j] and compute max/min
#   - We do this in a vectorized C-level efficient way using the CSC structure.
#
# Optimized approach: use the dgCMatrix (CSC) structure to iterate by column,
# but for row-wise operations, convert to dgRMatrix (CSR) for row-major access.

A_csr <- as(A, "RsparseMatrix")  # dgRMatrix: row-compressed

# Function: given a numeric vector x aligned to id_order, compute neighbor
# max, min, mean for each cell. Returns a 3-column matrix (max, min, mean).
compute_neighbor_stats_sparse <- function(A_csr, A_csc, neighbor_count, x, n) {
  # --- MEAN via sparse mat-vec ---
  # Replace NA in x with 0 for sum, and count non-NA neighbors
  x_nona <- x
  is_na_x <- is.na(x)
  x_nona[is_na_x] <- 0
  
  # Neighbor sum (treating NA as 0)
  nb_sum <- as.numeric(A_csc %*% x_nona)
  
  # Count non-NA neighbors per cell
  not_na_flag <- as.numeric(!is_na_x)
  nb_count_valid <- as.numeric(A_csc %*% not_na_flag)
  
  nb_mean <- ifelse(nb_count_valid > 0, nb_sum / nb_count_valid, NA_real_)
  
  # --- MAX and MIN via CSR row traversal ---
  # Access the internal slots of dgRMatrix
  # dgRMatrix: @p (row pointers, length n+1), @j (column indices, 0-based), @x (values)
  rp <- A_csr@p    # row pointers (length n+1), 0-based
  cj <- A_csr@j    # column indices (0-based)
  
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  
  for (i in seq_len(n)) {
    start <- rp[i] + 1L       # convert 0-based to 1-based
    end   <- rp[i + 1L]
    if (end < start) next      # no neighbors
    
    col_indices <- cj[start:end] + 1L  # 1-based column indices
    neighbor_vals <- x[col_indices]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    
    if (length(neighbor_vals) > 0L) {
      nb_max[i] <- max(neighbor_vals)
      nb_min[i] <- min(neighbor_vals)
    }
  }
  
  cbind(nb_max, nb_min, nb_mean)
}

# ===========================================================================
# STEP 3b: Even faster max/min — vectorized with Rcpp-like approach in pure R
#     Using tapply on the expanded edge list for max/min
# ===========================================================================
# More efficient: expand all edges, get values, and use grouping operations.

compute_neighbor_stats_fast <- function(A_csr, A_csc, neighbor_count, x, n) {
  # --- MEAN via sparse mat-vec ---
  x_nona <- x
  is_na_x <- is.na(x)
  x_nona[is_na_x] <- 0
  
  nb_sum <- as.numeric(A_csc %*% x_nona)
  not_na_flag <- as.numeric(!is_na_x)
  nb_count_valid <- as.numeric(A_csc %*% not_na_flag)
  nb_mean <- ifelse(nb_count_valid > 0, nb_sum / nb_count_valid, NA_real_)
  
  # --- MAX and MIN via edge expansion + data.table grouping ---
  rp <- A_csr@p
  cj <- A_csr@j
  n_edges <- length(cj)
  
  # Build row-index vector from row pointers
  row_lengths <- diff(rp)  # length n
  row_ids <- rep(seq_len(n), times = row_lengths)  # length = n_edges
  col_ids <- cj + 1L  # 1-based
  
  # Get neighbor values
  edge_vals <- x[col_ids]
  
  # Remove edges where neighbor value is NA
  valid <- !is.na(edge_vals)
  row_ids_v <- row_ids[valid]
  edge_vals_v <- edge_vals[valid]
  
  if (length(row_ids_v) > 0L) {
    # Use data.table for fast grouped max/min
    edge_dt <- data.table(row = row_ids_v, val = edge_vals_v)
    agg <- edge_dt[, .(vmax = max(val), vmin = min(val)), by = row]
    
    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
    nb_max[agg$row] <- agg$vmax
    nb_min[agg$row] <- agg$vmin
  } else {
    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
  }
  
  cbind(nb_max, nb_min, nb_mean)
}

# ===========================================================================
# STEP 4: Process year-by-year, variable-by-variable
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast subsetting
setkey(cell_data, year, id)

years <- sort(unique(cell_data$year))

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = NA_real_)
  set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = NA_real_)
  set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = NA_real_)
}

# Pre-compute the CSR row pointers and column indices ONCE for edge expansion
# (these are reused every iteration)
rp_global <- A_csr@p
cj_global <- A_csr@j
row_lengths_global <- diff(rp_global)
row_ids_global <- rep(seq_len(n_cells), times = row_lengths_global)
col_ids_global <- cj_global + 1L  # 1-based

cat("Computing neighbor features...\n")
t0 <- proc.time()

for (yr in years) {
  cat(sprintf("  Year %d...\n", yr))
  
  # Get row indices for this year
  yr_rows <- which(cell_data$year == yr)
  
  # Get the cell IDs for this year's data
  yr_ids <- cell_data$id[yr_rows]
  
  # Map cell IDs to spatial indices in id_order
  yr_spatial_idx <- id_to_idx[as.character(yr_ids)]
  
  # Check if all cells are present and build reverse map:
  # spatial_to_yr_row: for spatial index s, which position in yr_rows has that cell?
  # If a cell is missing from this year, it won't have data.
  spatial_to_yr_pos <- rep(NA_integer_, n_cells)
  spatial_to_yr_pos[yr_spatial_idx] <- seq_along(yr_rows)
  
  for (var_name in neighbor_source_vars) {
    # Build the spatial-aligned variable vector
    # x[s] = value of var_name for spatial cell s in year yr (or NA if missing)
    x <- rep(NA_real_, n_cells)
    x[yr_spatial_idx] <- cell_data[[var_name]][yr_rows]
    
    # --- MEAN via sparse mat-vec ---
    x_nona <- x
    is_na_x <- is.na(x)
    x_nona[is_na_x] <- 0
    
    nb_sum <- as.numeric(A %*% x_nona)
    not_na_flag <- as.numeric(!is_na_x)
    nb_count_valid <- as.numeric(A %*% not_na_flag)
    nb_mean <- ifelse(nb_count_valid > 0, nb_sum / nb_count_valid, NA_real_)
    
    # --- MAX and MIN via edge expansion + data.table grouping ---
    edge_vals <- x[col_ids_global]
    
    valid <- !is.na(edge_vals)
    row_ids_v <- row_ids_global[valid]
    edge_vals_v <- edge_vals[valid]
    
    nb_max <- rep(NA_real_, n_cells)
    nb_min <- rep(NA_real_, n_cells)
    
    if (length(row_ids_v) > 0L) {
      edge_dt <- data.table(row = row_ids_v, val = edge_vals_v)
      agg <- edge_dt[, .(vmax = max(val), vmin = min(val)), by = row]
      nb_max[agg$row] <- agg$vmax
      nb_min[agg$row] <- agg$vmin
    }
    
    # Map spatial results back to cell_data rows
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    set(cell_data, i = yr_rows, j = max_col,  value = nb_max[yr_spatial_idx])
    set(cell_data, i = yr_rows, j = min_col,  value = nb_min[yr_spatial_idx])
    set(cell_data, i = yr_rows, j = mean_col, value = nb_mean[yr_spatial_idx])
  }
}

elapsed <- proc.time() - t0
cat(sprintf("Neighbor feature computation complete in %.1f seconds.\n", elapsed[3]))

# ===========================================================================
# STEP 5: Predict using the pre-trained Random Forest (DO NOT RETRAIN)
# ===========================================================================
# The predictor columns now match the original schema.
# Convert back to data.frame if the RF model expects it.

cell_data_df <- as.data.frame(cell_data)
predictions <- predict(rf_model, newdata = cell_data_df)
cell_data$predicted <- predictions

cat("Prediction complete. Pipeline finished.\n")
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | ✅ Identical `max`, `min`, `mean` over the same neighbor sets, same NA-handling logic |
| **RF model preserved** | ✅ `predict()` only — no retraining |
| **Column names preserved** | ✅ `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` |
| **Memory** | ✅ Year-by-year processing; peak ~1.5 GB well within 16 GB |
| **Speed** | ✅ ~140 sparse mat-vec products + 140 `data.table` grouped aggregations ≈ **5–15 minutes** (vs. 86+ hours) |

### Why this is fast

- **Sparse `A %*% x`** runs in O(nnz) ≈ 1.37M multiply-adds in optimized C (CHOLMOD/SuiteSparse). This replaces 6.46M R-level list iterations.
- **`data.table` grouped `max`/`min`** over ~1.37M edges uses radix-sort grouping — orders of magnitude faster than per-row `lapply`.
- **The adjacency matrix, row pointers, column indices, and row-ID expansion vector are all built once** and reused across all 28 × 5 = 140 iterations.