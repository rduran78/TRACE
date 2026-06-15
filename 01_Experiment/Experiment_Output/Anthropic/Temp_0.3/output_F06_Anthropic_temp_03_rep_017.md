 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs string keys (`paste(id, year)`) for every neighbor–year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String hashing over 6.46M rows with ~4 neighbors each ≈ **25+ million string constructions and lookups**. The resulting list of 6.46M integer vectors also consumes substantial memory.

### Bottleneck B: `compute_neighbor_stats` — R-level `lapply` over 6.46M elements

For each of the 5 variables, `lapply` iterates over 6.46M list elements, subsetting a numeric vector and computing `max`, `min`, `mean`. That's **~32 million R-level function calls** (5 vars × 6.46M rows), each with overhead from anonymous function dispatch, subsetting, and NA handling.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular grid with a fixed rectangular kernel. Here the data is a **panel** (cell × year) stored as a long data frame, and the neighbor structure is an irregular `nb` object (not all cells have 4 neighbors — boundary cells, islands, masked cells). Focal operations would require reshaping into a 2D raster per year, running the focal, then reshaping back — feasible but fragile and risks misalignment. The better strategy is to **vectorize the neighbor computation directly** using sparse-matrix multiplication and grouped column operations.

---

## 2. Optimization Strategy

### Key Insight: Neighbor aggregation is a sparse matrix–vector product

If we construct a **sparse adjacency matrix W** of dimension `N_cells × N_cells` from the `nb` object, then for any year-slice of a variable `x`, the neighbor sum is `W %*% x`, the neighbor count is `W %*% (non-NA indicator)`, and the neighbor mean is their ratio. Max and min require a different approach but can be computed via a **grouped operation** on a long-form edge table.

### Plan

| Step | What | Speedup mechanism |
|------|------|-------------------|
| 1 | Build a sparse `N_cells × N_cells` rook adjacency matrix once | `Matrix::sparseMatrix` from `nb` — seconds |
| 2 | Reshape each variable into a `N_cells × N_years` matrix | Vectorized indexing — seconds |
| 3 | For **mean**: sparse matrix multiply `W %*% X` and divide by `W %*% (!is.na(X))` per year-column | Fully vectorized, C-level — seconds per variable |
| 4 | For **max** and **min**: build a long edge table `(from, to)`, join variable values, then `data.table` grouped `max`/`min` by `(from, year)` | `data.table` grouped ops — very fast |
| 5 | Map results back to the original `cell_data` row order | Vectorized merge |

**Expected runtime: ~2–5 minutes** instead of 86+ hours.

**Numerical equivalence**: The sparse-matrix mean is algebraically identical to the per-row `mean(neighbor_vals[!is.na()])`. The `data.table` grouped max/min are identical to per-row `max`/`min` with `na.rm = TRUE`. The trained Random Forest model is never touched.

---

## 3. Working R Code

```r
# ==============================================================================
# Prerequisites
# ==============================================================================
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix from nb object (once)
# ==============================================================================
build_sparse_adjacency <- function(nb_obj, id_order) {
  # nb_obj: list of integer vectors (indices into id_order), class "nb"
  # id_order: vector of cell IDs in the order used by the nb object
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove 0-neighbor placeholders (spdep uses integer(0) or 0)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  # Return matrix and the mapping from cell ID to matrix row index
  id_to_row <- setNames(seq_len(n), as.character(id_order))
  list(W = W, id_to_row = id_to_row, id_order = id_order)
}

adj <- build_sparse_adjacency(rook_neighbors_unique, id_order)
W          <- adj$W
id_to_row  <- adj$id_to_row
n_cells    <- length(id_order)

# ==============================================================================
# STEP 2: Build edge table for max/min (once)
# ==============================================================================
build_edge_dt <- function(W) {
  # Extract (i, j) pairs from sparse matrix
  W_t <- as(W, "TsparseMatrix")   # triplet form
  data.table(from = W_t@i + 1L, to = W_t@j + 1L)
}

edge_dt <- build_edge_dt(W)

# ==============================================================================
# STEP 3: Convert cell_data to data.table and establish index mappings
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# Ensure consistent year ordering
years     <- sort(unique(cell_dt$year))
n_years   <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# Map each row to (cell_matrix_row, year_column)
cell_dt[, cell_row := id_to_row[as.character(id)]]
cell_dt[, year_col := year_to_col[as.character(year)]]

# Row-order index for writing results back
cell_dt[, orig_idx := .I]

# ==============================================================================
# STEP 4: Function to compute neighbor stats for one variable
# ==============================================================================
compute_neighbor_features_fast <- function(cell_dt, var_name, W, edge_dt,
                                           n_cells, n_years, years) {
  
  # --- 4a. Reshape variable into N_cells x N_years matrix ---
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(cell_dt$cell_row, cell_dt$year_col)] <- cell_dt[[var_name]]
  
  # --- 4b. Neighbor MEAN via sparse matrix multiply ---
  # non-NA indicator
  notNA <- !is.na(X)
  storage.mode(notNA) <- "double"   # for matrix multiply
  
  # Sum of neighbor values (NA treated as 0 after masking)
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  
  neighbor_sum   <- as.matrix(W %*% X_zero)       # n_cells x n_years
  neighbor_count <- as.matrix(W %*% notNA)         # n_cells x n_years
  
  neighbor_mean <- neighbor_sum / neighbor_count   # NaN where count==0
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  # --- 4c. Neighbor MAX and MIN via edge table + data.table grouped ops ---
  # For each edge (from -> to), get the "to" cell's value per year.
  # We need to do this across all years simultaneously.
  
  # Build a long table: edge × year
  # Efficient approach: for each year column, look up values for all edges
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Process year by year (each year is fast: ~1.37M edges)
  from_vec <- edge_dt$from
  to_vec   <- edge_dt$to
  
  for (t in seq_len(n_years)) {
    vals_t <- X[to_vec, t]   # neighbor values for all edges in year t
    
    # Remove NAs before grouping
    valid <- !is.na(vals_t)
    if (!any(valid)) next
    
    dt_t <- data.table(from = from_vec[valid], val = vals_t[valid])
    
    agg <- dt_t[, .(mx = max(val), mn = min(val)), by = from]
    
    max_mat[agg$from, t] <- agg$mx
    min_mat[agg$from, t] <- agg$mn
  }
  
  # --- 4d. Map results back to cell_dt row order ---
  idx <- cbind(cell_dt$cell_row, cell_dt$year_col)
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := max_mat[idx]]
  cell_dt[, (min_col)  := min_mat[idx]]
  cell_dt[, (mean_col) := neighbor_mean[idx]]
  
  invisible(cell_dt)
}

# ==============================================================================
# STEP 5: Run for all 5 neighbor source variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_fast(
    cell_dt, var_name, W, edge_dt, n_cells, n_years, years
  )
}

# ==============================================================================
# STEP 6: Clean up helper columns and convert back if needed
# ==============================================================================
cell_dt[, c("cell_row", "year_col", "orig_idx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ==============================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | 6.46M string-key list construction | Sparse matrix built once from `nb` object |
| **Mean computation** | R-level `lapply` + `mean()` × 6.46M × 5 | Sparse matrix multiply (`W %*% X`), C-level |
| **Max/Min computation** | R-level `lapply` + `max()`/`min()` × 6.46M × 5 | `data.table` grouped aggregation per year (~28 iterations × 1.37M edges) |
| **Memory** | ~6.46M-element list of integer vectors | ~344K × 28 matrices + sparse matrix (~5.5M non-zeros) |
| **Estimated runtime** | 86+ hours | **2–5 minutes** |
| **Numerical result** | Baseline | **Identical** (same arithmetic, no approximation) |
| **RF model** | Pre-trained | **Untouched** |

The sparse-matrix mean is algebraically identical to summing non-NA neighbor values and dividing by the non-NA count. The `data.table` grouped `max`/`min` with NA removal produces the same result as the original per-row `max(neighbor_vals[!is.na()])`. No retraining of the Random Forest is performed.