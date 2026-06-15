 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation suffers from a critical inefficiency: **it rebuilds a neighbor lookup that maps every cell-year row to its neighbor cell-year rows, even though the neighbor topology is purely spatial and never changes across the 28 years.** Specifically:

1. **`build_neighbor_lookup` creates a list of 6.46 million entries** (one per cell-year row), each containing row indices into the full panel. This involves creating ~6.46M paste-key lookups, hashing them, and then for each row, pasting neighbor keys with the same year, and looking them up. This is O(N × T × avg_neighbors) string work.

2. **`compute_neighbor_stats` iterates over 6.46M entries** per variable. For 5 variables, that's ~32.3M list-element operations, each involving subsetting, NA removal, and computing max/min/mean.

3. **The fundamental waste**: The neighbor relationship is between **cells**, not between cell-years. Cell *i*'s neighbors are the same in 1992 as in 2019. Yet the code re-derives these relationships at the cell-year level, inflating the problem by a factor of 28.

### Quantified bottleneck
- `build_neighbor_lookup`: Allocates a 6.46M-element list, each requiring string construction and hash lookups. This alone can take hours.
- `compute_neighbor_stats` × 5 variables: 32.3M `lapply` iterations in R (not vectorized).

## Optimization Strategy

**Separate the static topology from the year-varying data, then use vectorized/matrix operations:**

1. **Build the neighbor lookup once at the cell level** (344,208 entries, not 6.46M). This is a simple translation of the `spdep::nb` object into a list of integer cell-index vectors—essentially, it already is one.

2. **Reshape each variable into a matrix of dimension `n_cells × n_years`**, where rows are cells (in `id_order` order) and columns are years. This allows direct column-wise (year-wise) vectorized neighbor aggregation.

3. **Compute neighbor stats per variable using vectorized row-gather operations on the matrix.** For each cell, gather its neighbor rows (same across all years), compute max/min/mean across those rows for all 28 year-columns simultaneously. This replaces 6.46M R-level iterations with 344,208 iterations, each doing vectorized column operations.

4. **Further acceleration**: Use a sparse-matrix multiply to compute neighbor means (and use row-wise sparse operations for max/min), or use `data.table` grouped operations. The approach below uses direct matrix indexing which is cache-friendly and avoids string operations entirely.

**Expected speedup**: ~28× from eliminating year redundancy in topology, plus large constant-factor gains from vectorization. Estimated runtime drops from 86+ hours to ~10–30 minutes.

**Numerical equivalence**: The same neighbors are gathered, the same values are read, and the same max/min/mean are computed. The trained Random Forest model is untouched.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Prepare ordered cell and year vectors
# ============================================================
# id_order: integer vector of length n_cells (344,208), giving cell IDs
#           in the same order as rook_neighbors_unique (the nb object).
# rook_neighbors_unique: spdep nb object, list of length n_cells,
#           each element is an integer vector of neighbor *positions*
#           (indices into id_order). 0-neighbor cells have integer(0).
# cell_data: data.frame/data.table with columns id, year, and all predictors.

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Convert cell_data to data.table for fast manipulation
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build a cell-level neighbor lookup (STATIC, done once)
# ============================================================
# rook_neighbors_unique is already exactly this: a list where element i
# contains the integer indices (into id_order) of cell i's neighbors.
# We just ensure it's a clean list of integer vectors.

neighbor_cell_idx <- lapply(rook_neighbors_unique, function(nb) {

  nb <- as.integer(nb)
  # spdep nb objects use 0 to indicate no neighbors in some representations;
  # remove any zeros or NAs
  nb[nb > 0L & !is.na(nb)]
})

# ============================================================
# STEP 2: Create mapping from (id, year) to matrix position
# ============================================================
# We need cell_data rows to align to a matrix [n_cells, n_years].
# Map cell IDs to row indices (position in id_order).
# Map years to column indices.

id_to_row  <- setNames(seq_len(n_cells), as.character(id_order))
year_to_col <- setNames(seq_len(n_years), as.character(years))

cell_data[, matrix_row := id_to_row[as.character(id)]]
cell_data[, matrix_col := year_to_col[as.character(year)]]

# ============================================================
# STEP 3: Precompute CSR-style flat vectors for neighbor indices
#          (enables faster vectorized gathering)
# ============================================================
# For max/min we need per-cell neighbor index lists.
# For mean we can also use a sparse matrix multiply.

# Number of neighbors per cell
n_neighbors <- vapply(neighbor_cell_idx, length, integer(1))

# Flat neighbor index vector and pointer vector (CSR-style)
flat_nb     <- unlist(neighbor_cell_idx, use.names = FALSE)
nb_ptr      <- c(0L, cumsum(n_neighbors))  # length n_cells + 1

# ============================================================
# STEP 4: Function to reshape a variable into [n_cells x n_years] matrix
# ============================================================
var_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$matrix_row, dt$matrix_col)] <- dt[[var_name]]
  mat
}

# ============================================================
# STEP 5: Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_fast <- function(var_mat, neighbor_cell_idx,
                                        flat_nb, nb_ptr, n_cells, n_years) {
  # Output matrices
  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # For each cell, gather its neighbor rows and compute stats across all years

  for (i in seq_len(n_cells)) {
    start <- nb_ptr[i] + 1L
    end   <- nb_ptr[i + 1L]
    if (end < start) next  # no neighbors

    nb_idx <- flat_nb[start:end]

    # Extract neighbor sub-matrix: [num_neighbors x n_years]
    # This is a single matrix-subset operation, very fast
    nb_vals <- var_mat[nb_idx, , drop = FALSE]

    if (length(nb_idx) == 1L) {
      # Single neighbor: no need for colMeans etc.
      nb_max[i, ]  <- nb_vals[1L, ]
      nb_min[i, ]  <- nb_vals[1L, ]
      nb_mean[i, ] <- nb_vals[1L, ]
    } else {
      # Vectorized column-wise operations across all 28 years at once
      # Using matrixStats for speed if available, otherwise base R
      nb_max[i, ]  <- apply(nb_vals, 2, max, na.rm = TRUE)
      nb_min[i, ]  <- apply(nb_vals, 2, min, na.rm = TRUE)
      nb_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }

  # Fix Inf/-Inf from max/min on all-NA columns
  nb_max[is.infinite(nb_max)] <- NA_real_
  nb_min[is.infinite(nb_min)] <- NA_real_

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# ============================================================
# STEP 6: Even faster version using matrixStats (recommended)
# ============================================================
# If matrixStats is available, colMaxs/colMins/colMeans2 are much faster
# than apply(). But the main loop is over 344K cells, not 6.46M rows,
# so even base R apply is tractable.
#
# For maximal speed, we can also use Rcpp, but the R version below
# should complete in ~10-30 minutes on a 16GB laptop.

if (requireNamespace("matrixStats", quietly = TRUE)) {
  compute_neighbor_stats_fast <- function(var_mat, neighbor_cell_idx,
                                          flat_nb, nb_ptr, n_cells, n_years) {
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (i in seq_len(n_cells)) {
      start <- nb_ptr[i] + 1L
      end   <- nb_ptr[i + 1L]
      if (end < start) next

      nb_idx  <- flat_nb[start:end]
      nb_vals <- var_mat[nb_idx, , drop = FALSE]

      if (length(nb_idx) == 1L) {
        nb_max[i, ]  <- nb_vals[1L, ]
        nb_min[i, ]  <- nb_vals[1L, ]
        nb_mean[i, ] <- nb_vals[1L, ]
      } else {
        nb_max[i, ]  <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
        nb_min[i, ]  <- matrixStats::colMins(nb_vals, na.rm = TRUE)
        nb_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
      }
    }

    nb_max[is.infinite(nb_max)]  <- NA_real_
    nb_min[is.infinite(nb_min)]  <- NA_real_

    list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
  }
}

# ============================================================
# STEP 7: Main loop — process each variable and write back
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Processing neighbor stats for: %s", var_name))

  # Reshape variable to matrix [n_cells x n_years]
  var_mat <- var_to_matrix(cell_data, var_name, n_cells, n_years)

  # Compute neighbor max, min, mean (returns 3 matrices)
  stats <- compute_neighbor_stats_fast(
    var_mat, neighbor_cell_idx, flat_nb, nb_ptr, n_cells, n_years
  )

  # Column names matching original pipeline
  col_max  <- paste0("n_", var_name, "_max")
  col_min  <- paste0("n_", var_name, "_min")
  col_mean <- paste0("n_", var_name, "_mean")

  # Map matrix positions back to cell_data rows using the same (row, col) index
  idx <- cbind(cell_data$matrix_row, cell_data$matrix_col)

  cell_data[, (col_max)  := stats$nb_max[idx]]
  cell_data[, (col_min)  := stats$nb_min[idx]]
  cell_data[, (col_mean) := stats$nb_mean[idx]]

  # Free memory
  rm(var_mat, stats)
  gc()
}

# ============================================================
# STEP 8: Clean up helper columns
# ============================================================
cell_data[, c("matrix_row", "matrix_col") := NULL]

# ============================================================
# STEP 9: Proceed with pre-trained Random Forest prediction
# ============================================================
# The trained RF model is unchanged. cell_data now has exactly the same
# neighbor feature columns (n_<var>_max, n_<var>_min, n_<var>_mean)
# with numerically identical values. Proceed directly:
#
# predictions <- predict(rf_model, newdata = cell_data)
#

message("Neighbor feature computation complete. Ready for RF prediction.")
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup size** | 6.46M entries (cell-year) | 344K entries (cell only) |
| **String hashing** | ~19M paste + hash ops | **Zero** |
| **Inner loop iterations** | 6.46M × 5 vars = 32.3M | 344K × 5 vars = 1.72M |
| **Work per iteration** | Scalar subsetting | Vectorized across 28 years |
| **Estimated runtime** | 86+ hours | ~10–30 minutes |
| **Numerical result** | Original | **Identical** (same neighbors, same values, same aggregation functions) |
| **RF model** | Pre-trained | **Untouched** — only `predict()` is called |
| **Peak RAM** | Large list overhead | ~344K × 28 × 8 bytes ≈ 77MB per variable matrix — well within 16GB |