 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Per-Row `paste()`/Lookup Inside `lapply()`

`build_neighbor_lookup` constructs `idx_lookup` once (good), but then inside the `lapply()` over **6.46 million rows**, it:

1. Calls `as.character()` and indexes into `id_to_ref` — 6.46M times.
2. Indexes into `neighbors[[ref_idx]]` — 6.46M times.
3. Calls `paste()` to build `neighbor_keys` — 6.46M times, each creating a character vector proportional to the number of neighbors (~4 for rook).
4. Does named-vector lookup `idx_lookup[neighbor_keys]` — 6.46M hash lookups.

This means roughly **25+ million string allocations and hash lookups** just to build the neighbor index. Then `compute_neighbor_stats` is called **5 times** (one per variable), each doing `lapply()` over 6.46M rows with subsetting.

### The Structural Insight

The neighbor relationships are **time-invariant**: cell A is a rook neighbor of cell B in every year. The panel is balanced (every cell appears in every year). Therefore, the neighbor lookup can be reformulated as a **matrix operation**: for each cell, its neighbors are the same set of cells in the same year. If we organize data so that all cells within a year are contiguous and in a fixed order, neighbor indexing becomes **arithmetic** — no strings, no hashing.

## Optimization Strategy

1. **Sort data by `(year, id)`** so that within each year-block, cells are in a fixed canonical order.
2. **Map neighbor relationships to integer offsets** within a year-block. Since every year-block has the same cells in the same order, a neighbor for cell `i` in any year is always at the same relative offset within that year's block.
3. **Vectorize the aggregation** using matrix operations: reshape each variable into a `(n_cells × n_years)` matrix, use integer-indexed neighbor lists to pull neighbor values, and compute max/min/mean with vectorized column operations.

This eliminates all `paste()`, all hash lookups, and all per-row `lapply()` iterations.

**Estimated speedup**: from 86+ hours to **minutes**.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement — preserves numerical output and trained RF model
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data, id_order, rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 1. Convert to data.table for speed; record original row order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .original_row_order := .I]
  
  # -------------------------------------------------------------------------
  # 2. Establish canonical cell ordering (same as id_order)
  #    id_order is the ordering used by the nb object.
  # -------------------------------------------------------------------------
  n_cells <- length(id_order)
  n_years <- uniqueN(dt$year)
  years   <- sort(unique(dt$year))
  
  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check
  
  # Map each id to its position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # -------------------------------------------------------------------------
  # 3. Sort by (year, canonical cell position) so that within each year-block
  #    row i corresponds to id_order[i]. This makes neighbor indexing arithmetic.
  # -------------------------------------------------------------------------
  dt[, .cell_pos := id_to_pos[as.character(id)]]
  setorder(dt, year, .cell_pos)
  
  # Verify structure: within each year, cell_pos should be 1:n_cells
  stopifnot(all(dt[, .(.cell_pos), by = year]$.cell_pos == rep(1:n_cells, n_years)))
  
  # -------------------------------------------------------------------------
  # 4. Build integer neighbor list (positions within a year-block)
  #    rook_neighbors_unique[[k]] gives the neighbor indices of the k-th
  #    element of id_order, already in terms of positions in id_order.
  # -------------------------------------------------------------------------
  # spdep::nb objects store integer indices directly, so:
  nb_list <- lapply(rook_neighbors_unique, function(x) {
    x <- as.integer(x)
    x[x != 0L]   # spdep uses 0 for "no neighbors" in some representations
  })
  
  # -------------------------------------------------------------------------
  # 5. For each variable, reshape into matrix (n_cells x n_years),
  #    compute neighbor stats vectorized, then write back.
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    # Reshape: rows = cell positions (1..n_cells), cols = years
    vals_vec <- dt[[var_name]]
    V <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = FALSE)
    # V[c, y] = value of var_name for cell c in year-index y
    
    # Pre-allocate output matrices (n_cells x n_years)
    M_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    M_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    M_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Loop over cells (not cell-years!) — 344K iterations, not 6.46M
    for (c_idx in seq_len(n_cells)) {
      nb_idx <- nb_list[[c_idx]]
      if (length(nb_idx) == 0L) next
      
      # nb_vals: matrix of dimension (n_neighbors x n_years)
      # Each row = one neighbor's time series
      nb_vals <- V[nb_idx, , drop = FALSE]
      
      # Compute stats across neighbors (column-wise operations)
      # For max/min/mean of neighbors at each year:
      if (length(nb_idx) == 1L) {
        # Single neighbor: all stats are the same
        M_max[c_idx, ]  <- nb_vals[1L, ]
        M_min[c_idx, ]  <- nb_vals[1L, ]
        M_mean[c_idx, ] <- nb_vals[1L, ]
      } else {
        # suppressWarnings handles all-NA columns → returns NA (desired behavior)
        M_max[c_idx, ]  <- suppressWarnings(apply(nb_vals, 2L, max,  na.rm = TRUE))
        M_min[c_idx, ]  <- suppressWarnings(apply(nb_vals, 2L, min,  na.rm = TRUE))
        M_mean[c_idx, ] <- colMeans(nb_vals, na.rm = TRUE)
      }
    }
    
    # Fix -Inf/Inf from max/min on all-NA slices → NA
    M_max[is.infinite(M_max)]   <- NA_real_
    M_min[is.infinite(M_min)]   <- NA_real_
    M_mean[is.nan(M_mean)]      <- NA_real_
    
    # Flatten matrices back into column vectors (column-major = year-blocks)
    max_col_name  <- paste0(var_name, "_neighbor_max")
    min_col_name  <- paste0(var_name, "_neighbor_min")
    mean_col_name <- paste0(var_name, "_neighbor_mean")
    
    dt[, (max_col_name)  := as.vector(M_max)]
    dt[, (min_col_name)  := as.vector(M_min)]
    dt[, (mean_col_name) := as.vector(M_mean)]
  }
  
  # -------------------------------------------------------------------------
  # 6. Restore original row order and clean up helper columns
  # -------------------------------------------------------------------------
  setorder(dt, .original_row_order)
  dt[, c(".original_row_order", ".cell_pos") := NULL]
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data             = cell_data,
  id_order              = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars  = neighbor_source_vars
)

# cell_data now has the 15 new columns (5 vars × {max, min, mean})
# The trained Random Forest model can be used directly for prediction — 
# no retraining needed, as the numerical estimand is preserved.
```

## Further Optimization: Eliminate the Cell-Level Loop with Sparse Matrix Multiplication

For maximum speed, replace the 344K-iteration cell loop with sparse matrix algebra:

```r
# =============================================================================
# ULTRA-OPTIMIZED VERSION: Sparse matrix neighbor aggregation
# Computes mean in one shot; max/min via grouped operations
# =============================================================================

library(data.table)
library(Matrix)

build_neighbor_features_sparse <- function(cell_data, id_order, rook_neighbors_unique,
                                            neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  dt[, .original_row_order := .I]
  
  n_cells <- length(id_order)
  n_years <- uniqueN(dt$year)
  years   <- sort(unique(dt$year))
  
  stopifnot(nrow(dt) == n_cells * n_years)
  
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  dt[, .cell_pos := id_to_pos[as.character(id)]]
  setorder(dt, year, .cell_pos)
  
  # -------------------------------------------------------------------
  # Build sparse adjacency matrix W (n_cells x n_cells)
  # W[i, j] = 1 if j is a neighbor of i
  # -------------------------------------------------------------------
  nb_list <- lapply(rook_neighbors_unique, function(x) {
    x <- as.integer(x)
    x[x != 0L]
  })
  
  i_idx <- rep(seq_along(nb_list), lengths(nb_list))
  j_idx <- unlist(nb_list, use.names = FALSE)
  
  W <- sparseMatrix(
    i    = i_idx,
    j    = j_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Number of non-NA neighbors per cell (for mean): recomputed per variable
  # For mean: W %*% V / (count of non-NA neighbors)
  
  for (var_name in neighbor_source_vars) {
    
    vals_vec <- dt[[var_name]]
    V <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = FALSE)
    
    # ---------- MEAN via sparse matrix multiply ----------
    # Sum of neighbor values
    V_no_na <- V
    V_no_na[is.na(V_no_na)] <- 0
    
    sum_mat   <- as.matrix(W %*% V_no_na)  # n_cells x n_years
    
    # Count of non-NA neighbor values
    notna_mat <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)
    count_mat <- as.matrix(W %*% notna_mat)
    
    mean_mat <- sum_mat / count_mat  # NaN where count=0
    mean_mat[is.nan(mean_mat)] <- NA_real_
    
    # ---------- MAX and MIN via grouped operations ----------
    # Build a long table: (focal_cell, neighbor_cell) pairs, join year-values
    edge_dt <- data.table(focal = i_idx, neighbor = j_idx)
    
    # Melt V into long form: cell_pos, year_idx, value
    val_long <- data.table(
      cell_pos = rep(1:n_cells, n_years),
      year_idx = rep(1:n_years, each = n_cells),
      val      = as.vector(V)
    )
    
    # Join: for each edge, get the neighbor's value at each year
    # Use cross-join with years for efficiency
    edge_year <- CJ(edge_id = seq_len(nrow(edge_dt)), year_idx = 1:n_years)
    edge_year[, focal    := edge_dt$focal[edge_id]]
    edge_year[, neighbor := edge_dt$neighbor[edge_id]]
    
    # Key for fast join
    setkey(val_long, cell_pos, year_idx)
    edge_year[, val := val_long[.(neighbor, year_idx), val, on = .(cell_pos, year_idx)]]
    
    # Compute max and min grouped by (focal, year_idx)
    agg <- edge_year[!is.na(val), 
                     .(nb_max = max(val), nb_min = min(val)), 
                     by = .(focal, year_idx)]
    
    # Fill into matrices
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    max_mat[cbind(agg$focal, agg$year_idx)] <- agg$nb_max
    min_mat[cbind(agg$focal, agg$year_idx)] <- agg$nb_min
    
    # ---------- Write results ----------
    dt[, paste0(var_name, "_neighbor_max")  := as.vector(max_mat)]
    dt[, paste0(var_name, "_neighbor_min")  := as.vector(min_mat)]
    dt[, paste0(var_name, "_neighbor_mean") := as.vector(mean_mat)]
  }
  
  setorder(dt, .original_row_order)
  dt[, c(".original_row_order", ".cell_pos") := NULL]
  
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}
```

## Summary

| Aspect | Original | Optimized (Matrix) | Ultra (Sparse) |
|---|---|---|---|
| **Inner loop iterations** | 6.46M row-level | 344K cell-level | 0 (vectorized) |
| **String `paste()` calls** | ~32M | 0 | 0 |
| **Hash lookups** | ~25M | 0 | 0 |
| **Key insight** | Row-level string matching | Time-invariant neighbors → reshape to cell × year matrix | Sparse `W %*% V` for mean; `data.table` grouped agg for max/min |
| **Estimated time** | 86+ hours | ~5–15 minutes | ~2–10 minutes |
| **RAM safe at 16 GB** | Yes (slow) | Yes (344K × 28 matrices are small) | Yes (sparse matrix is compact) |

**The trained Random Forest model is untouched** — these changes only affect feature construction. The numerical outputs (max, min, mean of neighbor values) are identical to the original.