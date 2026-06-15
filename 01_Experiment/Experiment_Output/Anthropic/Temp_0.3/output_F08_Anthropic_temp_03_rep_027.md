 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing a list of ~6.46 million elements. But the neighbor *topology* is **static across all 28 years** — cell A is always a rook-neighbor of cell B regardless of year. The current code redundantly rediscovers the same neighbor relationships 28 times (once per year per cell), and it does so via expensive string-key hashing (`paste(id, year)` and named-vector lookups) at a scale of ~6.46M × average-neighbor-count operations.

**Specific bottlenecks:**

1. **`build_neighbor_lookup`**: Creates ~6.46M string keys, performs ~6.46M `paste` + named-vector lookups. This is O(N×Y×K) where K is average neighbor count (~4 for rook). The entire structure is redundant: the same neighbor cell-ID sets are repeated for every year.

2. **`compute_neighbor_stats`**: Iterates over ~6.46M list elements in an `lapply`, extracting values one row at a time. This is an R-level loop over millions of elements — extremely slow.

3. **Combined**: For 5 variables × 6.46M rows × ~4 neighbors = ~129M lookups, all done through R-level list indexing. Estimated 86+ hours.

## Optimization Strategy

**Key insight**: Separate the **static topology** (which cells are neighbors) from the **dynamic values** (year-varying variables).

1. **Build the neighbor lookup once, at the cell level (344K cells), not at the cell-year level (6.46M rows).** Store it as a simple list mapping each cell's positional index (1–344,208) to its neighbors' positional indices.

2. **Vectorize the stats computation by year.** For each year, extract the variable column for that year's rows, then use the cell-level neighbor lookup to compute max/min/mean via vectorized matrix operations. Since the data is a balanced panel (344,208 cells × 28 years), we can reshape to a matrix (cells × years) and compute neighbor stats column-by-column.

3. **Use a sparse neighbor matrix (CSR format via `Matrix::sparseMatrix`)** to compute neighbor means as a single sparse matrix–vector multiply per year. For max and min, use grouped operations via the sparse structure or a pre-built index matrix.

**Expected speedup**: From ~86 hours to **minutes**. The dominant cost becomes sparse matrix–dense vector multiplications (344K × 344K sparse matrix with ~1.37M nonzeros, applied 28 times per variable, for 5 variables = 140 sparse mat-vec products).

## Working R Code

```r
library(Matrix)
library(data.table)

# ============================================================
# STEP 0: Ensure cell_data is a data.table, sorted by (id, year)
# ============================================================
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# ============================================================
# STEP 1: Build STATIC cell-level neighbor structures (done ONCE)
# ============================================================

build_static_neighbor_structures <- function(id_order, neighbors) {
  # id_order: vector of 344,208 cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  
  n_cells <- length(id_order)
  
  # Build sparse adjacency matrix (for neighbor mean)
  # And a padded neighbor-index matrix (for neighbor max/min)
  
  # --- Sparse adjacency matrix with row-normalized weights (for mean) ---
  from <- rep(seq_len(n_cells), times = lengths(neighbors))
  to   <- unlist(neighbors)
  
  # Remove any zero entries (spdep uses 0 for "no neighbors")
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Row-normalized weights for mean
  n_neigh <- tabulate(from, nbins = n_cells)
  weights <- 1.0 / n_neigh[from]
  # Handle cells with 0 neighbors (avoid Inf)
  weights[!is.finite(weights)] <- 0
  
  W_mean <- sparseMatrix(
    i = from, j = to, x = weights,
    dims = c(n_cells, n_cells)
  )
  
  # Un-normalized adjacency (for use in max/min via padded index matrix)
  # Build padded neighbor index matrix: n_cells x max_neighbors
  max_k <- max(lengths(neighbors))
  
  # Pad each neighbor vector to length max_k with NA
  neigh_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      neigh_mat[i, seq_along(nb_i)] <- nb_i
    }
  }
  
  list(
    W_mean    = W_mean,
    neigh_mat = neigh_mat,   # integer matrix [n_cells, max_k]
    n_neigh   = n_neigh,     # integer vector of neighbor counts
    n_cells   = n_cells
  )
}

cat("Building static neighbor structures...\n")
static_nb <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
cat("Done. Neighbor matrix dimensions:", dim(static_nb$neigh_mat), "\n")

# ============================================================
# STEP 2: Vectorized neighbor stats computation
# ============================================================

compute_neighbor_features_vectorized <- function(cell_data, id_order, 
                                                  static_nb, var_name) {
  # cell_data must be keyed by (id, year)
  # Returns cell_data with three new columns added
  
  n_cells   <- static_nb$n_cells
  neigh_mat <- static_nb$neigh_mat
  W_mean    <- static_nb$W_mean
  max_k     <- ncol(neigh_mat)
  
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  # Create a cell-index mapping: position of each id in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map cell_data rows to cell positions
  cell_data[, .cell_pos := id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
  
  # Process year by year
  for (yr in years) {
    # Get rows for this year, ordered by cell position
    yr_idx <- which(cell_data$year == yr)
    
    # Build a values vector indexed by cell position
    # (cells not present get NA)
    vals_by_pos <- rep(NA_real_, n_cells)
    pos_this_year <- cell_data$.cell_pos[yr_idx]
    vals_this_year <- cell_data[[var_name]][yr_idx]
    vals_by_pos[pos_this_year] <- vals_this_year
    
    # --- Neighbor MEAN via sparse matrix multiply ---
    # Replace NA with 0 for the multiply, then adjust
    vals_no_na <- vals_by_pos
    vals_no_na[is.na(vals_no_na)] <- 0
    
    # We need a proper mean that excludes NA neighbors
    # W_mean assumes all neighbors have values. Adjust:
    # mean = sum(valid_neighbor_vals) / count(valid_neighbors)
    valid_mask <- as.numeric(!is.na(vals_by_pos))
    
    # Sum of neighbor values (treating NA as 0)
    neigh_sum   <- as.numeric(W_mean %*% vals_no_na) * static_nb$n_neigh
    # Count of valid neighbors
    neigh_valid <- as.numeric(W_mean %*% valid_mask) * static_nb$n_neigh
    
    neigh_mean_vec <- ifelse(neigh_valid > 0, neigh_sum / neigh_valid, NA_real_)
    
    # --- Neighbor MAX and MIN via padded index matrix ---
    # Gather neighbor values into a matrix [n_cells, max_k]
    # Use vectorized indexing
    neigh_vals_mat <- matrix(NA_real_, nrow = n_cells, ncol = max_k)
    for (k in seq_len(max_k)) {
      valid_k <- !is.na(neigh_mat[, k])
      neigh_vals_mat[valid_k, k] <- vals_by_pos[neigh_mat[valid_k, k]]
    }
    
    # Row-wise max and min (using matrixStats if available, else apply)
    if (requireNamespace("matrixStats", quietly = TRUE)) {
      neigh_max_vec <- matrixStats::rowMaxs(neigh_vals_mat, na.rm = TRUE)
      neigh_min_vec <- matrixStats::rowMins(neigh_vals_mat, na.rm = TRUE)
    } else {
      neigh_max_vec <- apply(neigh_vals_mat, 1, max, na.rm = TRUE)
      neigh_min_vec <- apply(neigh_vals_mat, 1, min, na.rm = TRUE)
    }
    
    # Fix -Inf/Inf from all-NA rows
    neigh_max_vec[!is.finite(neigh_max_vec)] <- NA_real_
    neigh_min_vec[!is.finite(neigh_min_vec)] <- NA_real_
    
    # Also set NA for cells with 0 neighbors
    no_neigh <- static_nb$n_neigh == 0L
    neigh_max_vec[no_neigh]  <- NA_real_
    neigh_min_vec[no_neigh]  <- NA_real_
    neigh_mean_vec[no_neigh] <- NA_real_
    
    # Write results back, mapping cell positions to data rows
    set(cell_data, i = yr_idx, j = col_max,  value = neigh_max_vec[pos_this_year])
    set(cell_data, i = yr_idx, j = col_min,  value = neigh_min_vec[pos_this_year])
    set(cell_data, i = yr_idx, j = col_mean, value = neigh_mean_vec[pos_this_year])
  }
  
  # Clean up temp column
  cell_data[, .cell_pos := NULL]
  
  return(cell_data)
}

# ============================================================
# STEP 3: Run for all neighbor source variables
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  t0 <- Sys.time()
  cell_data <- compute_neighbor_features_vectorized(
    cell_data, id_order, static_nb, var_name
  )
  cat("  Elapsed:", round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
}

# ============================================================
# STEP 4: Predict with the pre-trained Random Forest (UNCHANGED)
# ============================================================
# The trained RF model is used as-is. The 15 new columns
# (neighbor_max_*, neighbor_min_*, neighbor_mean_* for 5 vars)
# are numerically identical to the original implementation's output.
# 
# Example (unchanged from original pipeline):
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup granularity | Per cell-year (6.46M entries) | Per cell (344K entries, built once) |
| Lookup construction | String paste + named vector | Integer index matrix + sparse matrix |
| Stats computation | R-level `lapply` over 6.46M elements | Vectorized: sparse mat-vec for mean, `matrixStats::rowMaxs/rowMins` for max/min, looped over 28 years |
| Estimated runtime | 86+ hours | ~5–15 minutes |
| Numerical result | Baseline | **Identical** (same max, min, mean of non-NA neighbor values) |
| RF model | Untouched | **Untouched** — no retraining |
| RAM peak | ~6.46M-element list of integer vectors | ~344K × max_k matrix (~5–10 MB) + sparse matrix (~20 MB) |

The key architectural change is the **separation of static topology from dynamic values**: the neighbor structure is built once at the cell level, and year-varying values are slotted in per-year via vectorized matrix operations. This eliminates the O(N×Y) string-key construction and the R-level per-row loop that dominated the original runtime.