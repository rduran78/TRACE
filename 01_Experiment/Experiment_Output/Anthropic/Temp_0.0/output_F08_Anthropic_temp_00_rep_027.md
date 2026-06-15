 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. This means:

1. **Redundant lookup construction**: The neighbor graph has only 344,208 cells, but the function builds 6.46M entries (344,208 × 28) by re-resolving the same neighbor cell IDs for every year. This is a 28× blowup in both time and memory.

2. **Redundant string hashing**: `paste(id, year, sep="_")` is called millions of times to create keys, and `idx_lookup[neighbor_keys]` performs millions of named-vector lookups (which are O(n) hash lookups on character vectors of length 6.46M).

3. **Row-level R `lapply` over 6.46M rows**: Each iteration does allocation, subsetting, `paste`, and NA checking — all in interpreted R. This is the dominant wall-clock cost.

4. **`compute_neighbor_stats` also loops over 6.46M entries**: Even though the neighbor *indices* within a given year are structurally the same for each cell, they are recomputed per row.

**In summary**: The static neighbor topology is entangled with the dynamic year dimension, causing a ~28× blowup in work and memory, compounded by slow interpreted-R loops over millions of rows.

## Optimization Strategy

**Separate the static topology from the dynamic data:**

1. **Build the neighbor lookup once, over cells only (344,208 entries)**, mapping each cell index to its neighbor cell indices. This is year-independent.

2. **For each variable, extract a matrix of values**: rows = cells, columns = years. This allows vectorized column-wise (year-wise) operations.

3. **Compute neighbor max/min/mean using vectorized matrix operations**: For each cell, gather neighbor rows from the matrix, then compute stats across neighbors for all years simultaneously. Better yet, use sparse-matrix or direct C++-level operations.

4. **Use `data.table` for fast reshaping** and avoid `paste`-based key lookups entirely.

5. **Optionally use a sparse adjacency matrix** to compute neighbor means as a matrix multiply, and neighbor max/min via row-wise sparse operations.

This reduces the problem from 6.46M interpreted-R iterations to ~344K iterations (or fully vectorized sparse-matrix operations), and eliminates all string-key hashing.

## Working R Code

```r
library(data.table)
library(Matrix)

# ===========================================================================
# STEP 1: Build the static cell-level neighbor lookup ONCE
#         (344,208 cells, not 6.46M cell-years)
# ===========================================================================

# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

# We need a mapping from cell ID -> position in id_order
# rook_neighbors_unique[[i]] gives neighbor positions for id_order[i]

# Build a sparse adjacency matrix (344208 x 344208) from the nb object.
# This is the static topology.

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: list of integer vectors (neighbor indices), length n
  # Returns a sparse logical/numeric matrix of dimension n x n
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(to) & to > 0
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# ===========================================================================
# STEP 2: Convert cell_data to data.table; create cell index and year index
# ===========================================================================

cell_dt <- as.data.table(cell_data)

# Create a mapping from cell ID to cell index (position in id_order)
id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cidx := id_to_cidx[as.character(id)]]

# Create ordered year vector and year index
years_vec <- sort(unique(cell_dt$year))
n_years   <- length(years_vec)
year_to_yidx <- setNames(seq_along(years_vec), as.character(years_vec))
cell_dt[, yidx := year_to_yidx[as.character(year)]]

# Ensure data is keyed for fast access
setkey(cell_dt, cidx, yidx)

# ===========================================================================
# STEP 3: Function to compute neighbor stats for one variable
#         using sparse matrix operations (fully vectorized)
# ===========================================================================

compute_neighbor_features_sparse <- function(dt, W, var_name, id_order,
                                              n_cells, years_vec, n_years) {
  # Build a cell x year matrix of the variable values
  # dt must have columns: cidx, yidx, and var_name
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(dt$cidx, dt$yidx)] <- dt[[var_name]]

  # --- Neighbor MEAN ---
  # W %*% val_mat gives sum of neighbor values for each cell x year

  # We need the count of non-NA neighbors per cell x year to get the mean
  not_na_mat <- matrix(0, nrow = n_cells, ncol = n_years)
  not_na_mat[cbind(dt$cidx, dt$yidx)] <- as.numeric(!is.na(dt[[var_name]]))

  # Replace NA with 0 for the sum computation
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0

  neighbor_sum   <- as.matrix(W %*% val_mat_zero)   # n_cells x n_years
  neighbor_count <- as.matrix(W %*% not_na_mat)      # n_cells x n_years

  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- Neighbor MAX and MIN ---
  # These cannot be done with simple matrix multiply.
  # We iterate over cells (344K iterations, not 6.46M).
  # Use the nb object directly for the neighbor list.

  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Pre-extract the neighbor list from the sparse matrix
  # (or reuse rook_neighbors_unique directly)
  # Using rook_neighbors_unique is fastest since it's already a list.

  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    # Extract neighbor rows from val_mat: each row is a neighbor, columns are years
    nb_vals <- val_mat[nb_idx, , drop = FALSE]  # k_neighbors x n_years
    # Columnwise max and min (suppress warnings for all-NA columns)
    neighbor_max[i, ] <- suppressWarnings(apply(nb_vals, 2, max, na.rm = TRUE))
    neighbor_min[i, ] <- suppressWarnings(apply(nb_vals, 2, min, na.rm = TRUE))
  }
  # Fix Inf/-Inf from all-NA slices
  neighbor_max[is.infinite(neighbor_max)] <- NA_real_
  neighbor_min[is.infinite(neighbor_min)] <- NA_real_

  # --- Write results back to dt ---
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := neighbor_max[cbind(cidx, yidx)]]
  dt[, (min_col)  := neighbor_min[cbind(cidx, yidx)]]
  dt[, (mean_col) := neighbor_mean[cbind(cidx, yidx)]]

  return(dt)
}

# ===========================================================================
# STEP 4: Further optimize MAX/MIN with chunked C-style vectorization
#         (avoid per-cell apply loop using vapply + direct indexing)
# ===========================================================================

# Faster version: pre-compute neighbor pointer arrays and use vectorized ops
compute_neighbor_features_fast <- function(dt, W, var_name,
                                            n_cells, n_years,
                                            nb_list) {
  # Build cell x year matrix
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(dt$cidx, dt$yidx)] <- dt[[var_name]]

  # --- MEAN via sparse matrix multiply ---
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0

  not_na <- matrix(0, nrow = n_cells, ncol = n_years)
  not_na[cbind(dt$cidx, dt$yidx)] <- as.numeric(!is.na(dt[[var_name]]))

  neighbor_sum   <- as.matrix(W %*% val_mat_zero)

  neighbor_count <- as.matrix(W %*% not_na)
  neighbor_mean  <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- MAX / MIN: vectorized over years, loop over cells ---
  # Pre-allocate
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Process in chunks to be cache-friendly
  chunk_size <- 10000L
  n_chunks <- ceiling(n_cells / chunk_size)

  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n_cells)

    for (i in i_start:i_end) {
      nb_idx <- nb_list[[i]]
      if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next

      if (length(nb_idx) == 1L) {
        neighbor_max[i, ] <- val_mat[nb_idx, ]
        neighbor_min[i, ] <- val_mat[nb_idx, ]
      } else {
        nb_block <- val_mat[nb_idx, , drop = FALSE]
        neighbor_max[i, ] <- colMaxs_na(nb_block)
        neighbor_min[i, ] <- colMins_na(nb_block)
      }
    }
  }

  # Write back
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := neighbor_max[cbind(cidx, yidx)]]
  dt[, (min_col)  := neighbor_min[cbind(cidx, yidx)]]
  dt[, (mean_col) := neighbor_mean[cbind(cidx, yidx)]]

  return(dt)
}

# Helper: column-wise max/min ignoring NAs (faster than apply)
colMaxs_na <- function(mat) {
  # mat: k x n_years
  result <- mat[1L, ]
  if (nrow(mat) > 1L) {
    for (r in 2L:nrow(mat)) {
      row_r <- mat[r, ]
      update <- !is.na(row_r) & (is.na(result) | row_r > result)
      result[update] <- row_r[update]
    }
  }
  result
}

colMins_na <- function(mat) {
  result <- mat[1L, ]
  if (nrow(mat) > 1L) {
    for (r in 2L:nrow(mat)) {
      row_r <- mat[r, ]
      update <- !is.na(row_r) & (is.na(result) | row_r < result)
      result[update] <- row_r[update]
    }
  }
  result
}

# ===========================================================================
# STEP 5: Run the pipeline
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")
cat("Cells:", n_cells, " Years:", n_years, " Rows:", nrow(cell_dt), "\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "...\n")
  t0 <- proc.time()

  cell_dt <- compute_neighbor_features_fast(
    dt       = cell_dt,
    W        = W,
    var_name = var_name,
    n_cells  = n_cells,
    n_years  = n_years,
    nb_list  = rook_neighbors_unique
  )

  elapsed <- (proc.time() - t0)[3]
  cat("    Done in", round(elapsed / 60, 1), "minutes.\n")
}

# ===========================================================================
# STEP 6: Clean up helper columns and convert back if needed
# ===========================================================================

cell_dt[, c("cidx", "yidx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ===========================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# predictions <- predict(rf_model, newdata = cell_data)
# (The RF model is not retrained; only the feature-engineering step is optimized.)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup size** | 6.46M entries (cell × year) | 344K entries (cell only) |
| **String key hashing** | ~25M+ `paste` + named-vector lookups | **Zero** — integer indexing only |
| **Neighbor mean** | R loop over 6.46M rows | Sparse matrix multiply (`W %*% V`), fully vectorized |
| **Neighbor max/min** | R loop over 6.46M rows | R loop over 344K cells, vectorized across 28 years per cell |
| **Memory** | 6.46M-element list of integer vectors + 6.46M-length character key vector | 344K × 28 dense matrix (~77 MB per variable) + sparse W (~22 MB) |
| **Estimated time** | 86+ hours | **~15–45 minutes** (depending on hardware) |
| **Numerical result** | — | **Identical** (same max, min, mean computed from same neighbor sets) |
| **RF model** | — | **Untouched** — no retraining |

The key insight: **the neighbor graph is a static spatial property; only the variable values change by year.** By separating topology (sparse matrix / nb list, built once) from data (cell × year matrices, one per variable), we eliminate the 28× redundancy and unlock vectorized linear algebra for the mean and vectorized column operations for max/min.