 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list elements**, each produced by an `lapply` call that performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named character vector lookup in R is O(n) per probe when the vector is large (6.46M entries), and doing this 6.46M times yields O(n²) behavior. This single function likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in R-level `lapply`**, computing max/min/mean on small vectors. While each operation is cheap, the R interpreter overhead across 6.46M iterations × 5 variables = ~32.3M iterations is substantial.

3. **Memory pressure**: Storing 6.46M list entries (each a small integer vector) has enormous R object overhead (~200+ bytes per list element), totaling several GB just for the lookup structure, before any computation begins.

### Root Cause Summary

The code treats this as a **row-level problem** (one list entry per cell-year), when it is actually a **cell-level graph problem replicated identically across 28 years**. The topology is year-invariant — the same 344,208 cells have the same ~1.37M rook edges every year. The lookup should be built at the cell level (344K entries) and the aggregation should exploit vectorized sparse matrix operations, not R-level loops over 6.46M rows.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once at the cell level** (344,208 × 344,208 with ~1.37M nonzero entries) from the `nb` object using `spdep::nb2listw` → `Matrix::sparseMatrix`, or directly.

2. **Reshape each variable into a matrix** of dimension (344,208 cells × 28 years), so that one sparse matrix multiplication `A %*% X` simultaneously computes the **sum** of neighbor values for all cells across all years.

3. **Compute neighbor counts** via `A %*% 1-matrix` (accounting for NA propagation) to derive the **mean**. Compute **max** and **min** via year-column-wise grouped operations using the sparse structure.

4. **Max and min cannot be computed via matrix multiplication**, so we use the CSC/CSR structure of the sparse matrix to do vectorized grouped operations (essentially `rowmax` and `rowmin` over sparse columns) — still at the 344K-cell level, looped over 28 years, which is trivially fast.

5. **Result**: The dominant cost becomes sparse matrix–dense matrix multiplication (~1.37M edges × 28 years × 5 variables), which `Matrix` handles in optimized C code in seconds, not hours.

**Expected speedup**: From 86+ hours to **under 5 minutes** on a 16 GB laptop.

---

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original max, min, mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Convert cell_data to data.table for speed --------------------
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec, 
#   pop_density, def, usd_est_n2, and ~110 predictor columns.
# Assumes rook_neighbors_unique is an nb object (list of integer vectors)
#   indexed in the same order as id_order.
# Assumes id_order is the vector of cell IDs corresponding to nb indices.

cell_dt <- as.data.table(cell_data)

# ---- Step 1: Build cell-level sparse adjacency matrix (once) ---------------

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer neighbor index vectors)
  # n: number of spatial cells (length of nb_obj)
  # Returns: n x n sparse Matrix (dgCMatrix) with 1s at neighbor positions
  
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove any 0-length or invalid entries
  valid <- !is.na(to) & to >= 1L & to <= n
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

cat(sprintf("Adjacency matrix: %d x %d, %d nonzeros\n", 
            nrow(A), ncol(A), nnzero(A)))

# ---- Step 2: Create cell-index and year-index mappings ---------------------

# Map cell id -> integer index (1..n_cells) matching id_order / nb ordering
cell_id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Sorted unique years
years <- sort(unique(cell_dt$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# Assign each row its cell index and year column
cell_dt[, cell_idx := cell_id_to_idx[as.character(id)]]
cell_dt[, year_col := year_to_col[as.character(year)]]

# ---- Step 3: Function to reshape a variable into (n_cells x n_years) matrix

reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Returns a dense matrix; NA where cell-year is missing
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- dt[[var_name]]
  idx  <- cbind(dt$cell_idx, dt$year_col)
  mat[idx] <- vals
  mat
}

# ---- Step 4: Compute neighbor stats using sparse matrix operations ---------

compute_neighbor_stats_sparse <- function(A, X) {
  # A: n x n sparse adjacency matrix
  # X: n x T dense matrix of variable values (may contain NAs)
  # Returns: list with max_mat, min_mat, mean_mat (each n x T)
  
  n <- nrow(X)
  n_t <- ncol(X)
  
  # --- Mean via sparse matrix multiplication ---
  # Replace NA with 0 for summation, track non-NA counts
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  
  # Indicator of non-NA
  X_valid <- matrix(1, nrow = n, ncol = n_t)
  X_valid[is.na(X)] <- 0
  
  # Sum of neighbor values (NAs treated as 0)
  sum_mat   <- A %*% X_nona        # n x T, sparse %*% dense -> dense
  # Count of non-NA neighbors
  count_mat <- A %*% X_valid       # n x T
  
  # Convert from Matrix class to base matrix
  sum_mat   <- as.matrix(sum_mat)
  count_mat <- as.matrix(count_mat)
  
  # Mean: sum / count, NA where count == 0
  mean_mat <- sum_mat / count_mat
  mean_mat[count_mat == 0] <- NA_real_
  
  # --- Max and Min via CSR traversal ---
  # Convert A to row-compressed form for efficient row-wise neighbor access
  A_r <- as(A, "RsparseMatrix")  # dgRMatrix: row-compressed
  
  max_mat <- matrix(NA_real_, nrow = n, ncol = n_t)
  min_mat <- matrix(NA_real_, nrow = n, ncol = n_t)
  
  # A_r@j is 0-based column index, A_r@p is row pointer (0-based)
  row_ptr <- A_r@p   # length n+1
  col_idx <- A_r@j   # 0-based
  
  # Process each year column (28 iterations — very fast)
  for (t in seq_len(n_t)) {
    x_col <- X[, t]  # values for this year
    
    for (i in seq_len(n)) {
      start <- row_ptr[i] + 1L    # R 1-based
      end   <- row_ptr[i + 1L]    # R 1-based end
      
      if (end < start) next  # no neighbors
      
      nb_indices <- col_idx[start:end] + 1L  # convert 0-based to 1-based
      nb_vals <- x_col[nb_indices]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      
      if (length(nb_vals) > 0L) {
        max_mat[i, t] <- max(nb_vals)
        min_mat[i, t] <- min(nb_vals)
      }
    }
  }
  
  list(max_mat = max_mat, min_mat = min_mat, mean_mat = mean_mat)
}

# ---- Step 4b: Even faster max/min using vectorized approach ----------------
# The inner double loop (344K × 28) in pure R is ~9.6M iterations, which
# may take a few minutes. We can vectorize by processing per-year as vector ops.

compute_neighbor_stats_sparse_fast <- function(A, X) {
  n <- nrow(X)
  n_t <- ncol(X)
  
  # --- Mean ---
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  X_valid <- matrix(1, nrow = n, ncol = n_t)
  X_valid[is.na(X)] <- 0
  
  sum_mat   <- as.matrix(A %*% X_nona)
  count_mat <- as.matrix(A %*% X_valid)
  mean_mat  <- sum_mat / count_mat
  mean_mat[count_mat == 0] <- NA_real_
  
  # --- Max and Min ---
  # Strategy: For max, replace NA with -Inf, multiply by adjacency,
  # but standard matmul gives sum, not max. We must iterate over the 
  # sparse structure. Use dgCMatrix (CSC) and process column-by-column 
  # of X, leveraging the fact that A's structure is fixed.
  
  # Convert to triplet form once
  A_t <- as(A, "TsparseMatrix")  # dgTMatrix: i, j are 0-based
  from_vec <- A_t@i + 1L  # row indices (1-based) = "to" node receiving aggregation
  to_vec   <- A_t@j + 1L  # col indices (1-based) = "from" node (neighbor)
  n_edges  <- length(from_vec)
  
  max_mat <- matrix(NA_real_, nrow = n, ncol = n_t)
  min_mat <- matrix(NA_real_, nrow = n, ncol = n_t)
  
  for (t in seq_len(n_t)) {
    x_col <- X[, t]
    
    # Get neighbor values for all edges at once
    nb_vals <- x_col[to_vec]  # length = n_edges
    
    # For max: use -Inf for NA so they don't affect max
    nb_max <- nb_vals
    nb_max[is.na(nb_max)] <- -Inf
    
    # For min: use +Inf for NA so they don't affect min
    nb_min <- nb_vals
    nb_min[is.na(nb_min)] <- Inf
    
    # Grouped max/min by 'from_vec' (the node receiving the aggregation)
    # Use data.table for fast grouped operations
    edge_dt <- data.table(
      node = from_vec,
      val_max = nb_max,
      val_min = nb_min,
      is_valid = !is.na(nb_vals)
    )
    
    agg <- edge_dt[, .(
      mx = max(val_max),
      mn = min(val_min),
      any_valid = any(is_valid)
    ), by = node]
    
    # Assign results
    valid_rows <- agg$any_valid
    max_mat[agg$node[valid_rows], t] <- agg$mx[valid_rows]
    min_mat[agg$node[valid_rows], t] <- agg$mn[valid_rows]
    # Nodes with all-NA neighbors remain NA (already initialized)
  }
  
  list(max_mat = max_mat, min_mat = min_mat, mean_mat = mean_mat)
}

# ---- Step 5: Run the pipeline for all 5 neighbor source variables ----------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-extract triplet form for reuse (avoids repeated conversion)
A_t <- as(A, "TsparseMatrix")
from_vec <- A_t@i + 1L
to_vec   <- A_t@j + 1L
n_edges  <- length(from_vec)
rm(A_t)

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor features for: %s\n", var_name))
  t0 <- proc.time()
  
  # Reshape to cell x year matrix
  X <- reshape_to_matrix(cell_dt, var_name, n_cells, n_years)
  
  # --- Compute mean via sparse matmul ---
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  X_valid <- matrix(1, nrow = n_cells, ncol = n_years)
  X_valid[is.na(X)] <- 0
  
  sum_mat   <- as.matrix(A %*% X_nona)
  count_mat <- as.matrix(A %*% X_valid)
  mean_mat  <- sum_mat / count_mat
  mean_mat[count_mat == 0] <- NA_real_
  
  rm(X_nona, X_valid, sum_mat, count_mat)
  
  # --- Compute max and min via vectorized grouped operations ---
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (t in seq_len(n_years)) {
    x_col <- X[, t]
    nb_vals <- x_col[to_vec]  # neighbor values for all edges
    
    not_na <- !is.na(nb_vals)
    if (!any(not_na)) next
    
    # Filter to valid edges only
    f_node  <- from_vec[not_na]
    f_val   <- nb_vals[not_na]
    
    # Fast grouped max/min using data.table
    edge_dt <- data.table(node = f_node, val = f_val)
    
    agg_max <- edge_dt[, .(v = max(val)), by = node]
    agg_min <- edge_dt[, .(v = min(val)), by = node]
    
    max_mat[agg_max$node, t] <- agg_max$v
    min_mat[agg_min$node, t] <- agg_min$v
  }
  
  rm(X)
  
  # --- Write results back to cell_dt ---
  # Flatten matrices back to the row order of cell_dt using cell_idx, year_col
  idx_mat <- cbind(cell_dt$cell_idx, cell_dt$year_col)
  
  max_col_name  <- paste0("neighbor_max_", var_name)
  min_col_name  <- paste0("neighbor_min_", var_name)
  mean_col_name <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col_name)  := max_mat[idx_mat]]
  cell_dt[, (min_col_name)  := min_mat[idx_mat]]
  cell_dt[, (mean_col_name) := mean_mat[idx_mat]]
  
  rm(max_mat, min_mat, mean_mat)
  gc()
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Done in %.1f seconds\n", elapsed))
}

# ---- Step 6: Clean up helper columns and convert back if needed ------------
cell_dt[, c("cell_idx", "year_col") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ---- Step 7: Apply pre-trained Random Forest (unchanged) -------------------
# The trained RF model is preserved exactly as-is. The neighbor features
# are numerically equivalent to the original implementation.
#
# Example (assuming rf_model is the pre-trained model):
# predictions <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete. Neighbor features added. RF model preserved.\n")
```

---

## Why This Preserves Numerical Equivalence

| Statistic | Original method | Optimized method | Equivalence |
|-----------|----------------|-----------------|-------------|
| **Mean** | `mean(neighbor_vals[!is.na(...)])` | `sum(non-NA values) / count(non-NA values)` via `A %*% X_nona / A %*% X_valid` | Exact — identical arithmetic |
| **Max** | `max(neighbor_vals[!is.na(...)])` | `max(val)` grouped by node from edge list, NA edges excluded | Exact — same `max()` over same values |
| **Min** | `min(neighbor_vals[!is.na(...)])` | `min(val)` grouped by node from edge list, NA edges excluded | Exact — same `min()` over same values |
| **NA handling** | Returns `c(NA, NA, NA)` when no valid neighbors | Matrix initialized to `NA`, only written when valid neighbors exist | Exact |

The adjacency matrix `A` encodes exactly the same directed edges as `rook_neighbors_unique`. The triplet `(from_vec, to_vec)` traverses exactly the same neighbor pairs. No approximation is introduced.

---

## Performance Estimate

| Component | Original | Optimized |
|-----------|----------|-----------|
| Build lookup | ~hours (6.46M string ops) | ~2 sec (sparse matrix construction) |
| Mean (per var) | ~hours (6.46M R-level loops) | ~5 sec (sparse matmul in C) |
| Max/Min (per var) | included above | ~30 sec (28 × vectorized data.table groupby on ~1.37M edges) |
| **Total (5 vars)** | **86+ hours** | **~3–5 minutes** |
| Peak RAM | ~8–12 GB (list overhead) | ~4–6 GB (dense matrices + sparse matrix) |