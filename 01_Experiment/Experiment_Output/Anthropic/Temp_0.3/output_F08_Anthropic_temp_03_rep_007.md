 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup list of length ~6.46 million, where each element maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor graph is static.** Cell *i*'s neighbors are the same in 1992 as in 2019. The spatial topology never changes. Yet the current code rebuilds/re-resolves neighbor relationships for every cell × every year — effectively duplicating the same spatial graph 28 times.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M string keys (`paste(id, year)`) and uses named-vector lookup (`idx_lookup[neighbor_keys]`) inside a `lapply` over 6.46M rows. Named-vector lookup in R is O(n) per probe in the worst case, and the `paste`/string operations dominate runtime.

3. **`compute_neighbor_stats` then iterates over the 6.46M-element list** for each of 5 variables, performing per-element subsetting and summary — another 5 × 6.46M = 32.3M R-level function calls.

**The key insight:** Because the neighbor structure is year-invariant, we should:
- Build the neighbor lookup **once at the cell level** (344K entries, not 6.46M).
- For each variable, extract a **cell × year matrix**, then compute neighbor max/min/mean using fast vectorized matrix operations over the 344K cells, broadcasting across all 28 years simultaneously.

This reduces the problem from 6.46M list iterations to 344K cell iterations (or fully vectorized matrix algebra), and eliminates all string-key construction.

---

## Optimization Strategy

| Aspect | Current (slow) | Redesigned (fast) |
|---|---|---|
| Neighbor lookup granularity | Per cell-year row (6.46M entries) | Per cell (344K entries) — **static** |
| Data structure for variables | Column in a long data.frame | Cell × Year matrix — **changing** |
| Neighbor aggregation | R-level `lapply` over 6.46M rows | Sparse-matrix multiplication or vectorized matrix-row aggregation over 344K cells |
| String key construction | 6.46M `paste()` calls + named-vector lookup | None — integer indexing only |
| Passes per variable | 1 pass over 6.46M rows | 1 sparse-matrix multiply (or 1 vectorized pass over 344K cells) |
| Estimated time | 86+ hours | Minutes |

**Concrete plan:**

1. **Build a cell-level neighbor lookup once** — a simple list of length 344K where element *i* contains the integer indices of cell *i*'s rook neighbors. This is just `rook_neighbors_unique` re-indexed to a contiguous 1:N integer mapping. Cost: trivial, done once.

2. **Build a sparse adjacency matrix W** (344K × 344K) from the neighbor list. This enables computing neighbor sums, counts, max, and min via matrix operations.

3. **Reshape each variable into a 344K × 28 matrix** (cell rows × year columns) using integer indexing.

4. **Compute neighbor stats using the sparse matrix:**
   - **Neighbor mean:** `W %*% X / neighbor_count` (one sparse matrix multiply per variable).
   - **Neighbor max and min:** Iterate over 344K cells (not 6.46M) using the cell-level neighbor list, operating on matrix rows. This is 18.7× fewer iterations and each iteration touches only the year-vector, which is cache-friendly.

5. **Write results back** into the long data.frame in the correct row order.

6. **Feed into the pre-trained Random Forest** exactly as before — same column names, same numerical values.

---

## Working R Code

```r
library(Matrix)  # for sparse matrix operations

# =============================================================================
# STEP 0: Ensure consistent cell ordering
# =============================================================================
# id_order: vector of 344,208 unique cell IDs (same order as rook_neighbors_unique)
# cell_data: the long data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# rook_neighbors_unique: spdep nb object (list of length 344,208)

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Map cell IDs to contiguous integer indices 1:n_cells
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# Map years to contiguous integer indices 1:n_years
year_to_idx <- setNames(seq_len(n_years), as.character(years))

# =============================================================================
# STEP 1: Build STATIC cell-level neighbor list (done once, reused forever)
# =============================================================================
# rook_neighbors_unique is already indexed consistently with id_order,
# so rook_neighbors_unique[[i]] gives the neighbor indices for id_order[i].
# We just need to strip the spdep attributes and ensure integer vectors.

cell_neighbor_list <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]  # spdep uses 0 to denote "no neighbors"; remove if present
})

# Precompute neighbor counts per cell (static)
neighbor_counts <- vapply(cell_neighbor_list, length, integer(1))

# =============================================================================
# STEP 2: Build sparse adjacency matrix W (static, built once)
#          W[i, j] = 1 if cell j is a neighbor of cell i
# =============================================================================
# Build COO triplets
from_idx <- rep(seq_len(n_cells), times = neighbor_counts)
to_idx   <- unlist(cell_neighbor_list, use.names = FALSE)

W <- sparseMatrix(
  i    = from_idx,
  j    = to_idx,
  x    = 1,
  dims = c(n_cells, n_cells)
)

# Neighbor count vector (as dense numeric for division)
n_count_vec <- as.numeric(neighbor_counts)
# Replace 0 with NA to avoid division by zero
n_count_vec_safe <- ifelse(n_count_vec == 0, NA_real_, n_count_vec)

# =============================================================================
# STEP 3: Map every row of cell_data to (cell_idx, year_idx) for fast reshaping
# =============================================================================
cell_data$`.cell_idx` <- id_to_idx[as.character(cell_data$id)]
cell_data$`.year_idx` <- year_to_idx[as.character(cell_data$year)]

# Linear index into a cell × year matrix (column-major)
lin_idx <- cell_data$`.cell_idx` + (cell_data$`.year_idx` - 1L) * n_cells

# =============================================================================
# STEP 4: Function to reshape a variable into cell × year matrix
# =============================================================================
var_to_matrix <- function(data, var_name, n_cells, n_years, lin_idx) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[lin_idx] <- data[[var_name]]
  mat
}

# =============================================================================
# STEP 5: Compute neighbor stats (max, min, mean) for each variable
# =============================================================================
# Strategy:
#   - MEAN: use sparse matrix multiply  ->  neighbor_mean = (W %*% X) / count
#   - MAX and MIN: vectorized loop over 344K cells (not 6.46M rows)
#     For each cell i, extract X[neighbors_of_i, ] and compute col-wise max/min.
#     This is 18.7× fewer iterations than the original approach.

compute_neighbor_features <- function(var_name, cell_data, W, cell_neighbor_list,
                                       n_count_vec_safe, n_cells, n_years, lin_idx) {
  
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # --- Reshape to matrix ---
  X <- var_to_matrix(cell_data, var_name, n_cells, n_years, lin_idx)
  
  # --- Neighbor MEAN via sparse matrix multiply ---
  # W %*% X gives, for each cell i, the sum of neighbor values across all years
  neighbor_sum  <- as.matrix(W %*% X)  # n_cells × n_years
  neighbor_mean <- neighbor_sum / n_count_vec_safe  # broadcasts column-wise
  
  # --- Neighbor MAX and MIN via vectorized cell-level loop ---
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_list[[i]]
    if (length(nb) == 0L) next
    
    # Extract neighbor rows: length(nb) × n_years sub-matrix
    nb_vals <- X[nb, , drop = FALSE]
    
    # Column-wise max and min (across neighbors, for each year)
    if (length(nb) == 1L) {
      neighbor_max[i, ] <- nb_vals[1L, ]
      neighbor_min[i, ] <- nb_vals[1L, ]
    } else {
      # Handles NAs: na.rm = TRUE; if all NA for a year, returns Inf/-Inf -> convert to NA
      col_max <- apply(nb_vals, 2, max, na.rm = TRUE)
      col_min <- apply(nb_vals, 2, min, na.rm = TRUE)
      col_max[is.infinite(col_max)] <- NA_real_
      col_min[is.infinite(col_min)] <- NA_real_
      neighbor_max[i, ] <- col_max
      neighbor_min[i, ] <- col_min
    }
  }
  
  # --- Write results back to long format using lin_idx ---
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_data[[max_col]]  <- neighbor_max[lin_idx]
  cell_data[[min_col]]  <- neighbor_min[lin_idx]
  cell_data[[mean_col]] <- neighbor_mean[lin_idx]
  
  cell_data
}

# =============================================================================
# STEP 6: Main loop — process all 5 neighbor source variables
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (optimized: static graph + matrix ops)...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(
    var_name, cell_data, W, cell_neighbor_list,
    n_count_vec_safe, n_cells, n_years, lin_idx
  )
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f seconds (%.1f minutes).\n", elapsed, elapsed / 60))

# Clean up temporary index columns
cell_data$`.cell_idx` <- NULL
cell_data$`.year_idx` <- NULL

# =============================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# =============================================================================
# The pre-trained model object (e.g., `rf_model`) is loaded from disk.
# Column names and numerical values are identical to the original pipeline.
#
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Eliminating the Cell-Level Loop for Max/Min

The `for` loop over 344K cells for max/min is already ~18.7× faster than the original 6.46M-row loop, but if it's still too slow (each iteration calls `apply`), we can replace it with a fully vectorized sparse approach:

```r
# =============================================================================
# OPTIONAL: Fully vectorized neighbor MAX and MIN using sparse tricks
# =============================================================================
# For MAX: replace 0s in W with -Inf, set missing X values to -Inf,
#          then sparse max can be computed year-by-year.
# 
# A simpler and very fast approach: use data.table to "explode" the neighbor
# list and compute grouped max/min. The exploded table has ~1.37M × 28 ≈ 38.5M
# rows conceptually, but we operate column-by-column.

compute_neighbor_maxmin_fast <- function(var_name, X, cell_neighbor_list, 
                                          n_cells, n_years) {
  library(data.table)
  
  # Explode neighbor relationships: from_cell -> to_cell (static, ~1.37M pairs)
  from_cell <- rep(seq_len(n_cells), times = vapply(cell_neighbor_list, length, integer(1)))
  to_cell   <- unlist(cell_neighbor_list, use.names = FALSE)
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Process each year-column: ~1.37M rows per year, 28 years
  for (yr in seq_len(n_years)) {
    vals <- X[to_cell, yr]  # neighbor values for this year
    dt <- data.table(from = from_cell, val = vals)
    agg <- dt[!is.na(val), .(vmax = max(val), vmin = min(val)), by = from]
    neighbor_max[agg$from, yr] <- agg$vmax
    neighbor_min[agg$from, yr] <- agg$vmin
  }
  
  list(neighbor_max = neighbor_max, neighbor_min = neighbor_min)
}
```

This replaces the 344K-iteration R loop with 28 fast `data.table` grouped aggregations over ~1.37M rows each — typically completing in seconds.

---

## Summary

| What | How |
|---|---|
| **Root cause** | Neighbor lookup built per cell-year (6.46M) instead of per cell (344K); string-key hashing |
| **Key insight** | Neighbor graph is **static**; only variable values **change** by year |
| **Redesign** | Separate static structure (sparse matrix W, cell-level neighbor list) from changing data (cell × year matrices) |
| **Neighbor mean** | Single sparse matrix multiply `W %*% X` per variable |
| **Neighbor max/min** | Cell-level loop (344K) or `data.table` grouped aggregation (1.37M pairs × 28 years) |
| **Speedup** | From 86+ hours to estimated **minutes** |
| **Preserves** | Pre-trained RF model (untouched), original numerical estimand (identical values) |