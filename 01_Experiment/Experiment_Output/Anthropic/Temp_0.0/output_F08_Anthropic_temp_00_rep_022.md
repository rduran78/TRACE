 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this identically for each of the 28 years a cell appears. This means:

1. **Redundant topology computation**: The neighbor graph has ~344K cells and ~1.37M directed edges. This is year-invariant. Yet `build_neighbor_lookup` expands it to ~6.46M list entries (344K × 28), each containing the same neighbor indices re-derived via string-key lookups. This alone creates ~38 million string-paste-and-match operations.

2. **Redundant per-variable iteration**: `compute_neighbor_stats` then iterates over all ~6.46M list entries per variable (×5 variables = ~32.3M list traversals), when it could operate on a year-sliced matrix.

3. **Memory bloat**: The 6.46M-element list of integer vectors consumes substantial RAM and causes GC pressure on a 16 GB laptop.

**Root cause summary**: The static neighbor topology is entangled with the dynamic year dimension, causing an O(cells × years) expansion of what should be an O(cells) structure, multiplied again by O(variables).

## Optimization Strategy

**Separate the static topology from the dynamic variable values:**

1. **Build the neighbor lookup once over cells only** (~344K entries, not ~6.46M). Each entry maps a cell to its neighbor cells by positional index into `id_order`. This is year-invariant and built once.

2. **For each variable, extract a cells × years matrix** where row *i* corresponds to `id_order[i]` and columns correspond to years. This is a simple reshape.

3. **Compute neighbor stats via vectorized matrix operations**: For each cell *i* with neighbor set *N(i)*, extract the sub-matrix of neighbor values (rows = *N(i)*, columns = all years), then compute column-wise (i.e., per-year) max, min, mean. This processes all 28 years simultaneously per cell.

4. **Reshape results back** to the long cell-year format and attach to `cell_data`.

This reduces the lookup from ~6.46M entries to ~344K, eliminates all string-key operations, and replaces millions of R-level `lapply` iterations with vectorized matrix column operations. Expected speedup: roughly 20–50× (from 86+ hours to 2–4 hours or less).

**Numerical equivalence**: The same neighbor sets and the same max/min/mean aggregations are computed, just reorganized. The trained Random Forest model is untouched.

## Working R Code

```r
# =============================================================================
# STEP 1: Build a cell-only neighbor lookup (year-invariant, built ONCE)
# =============================================================================
build_cell_neighbor_lookup <- function(id_order, rook_neighbors) {

  # rook_neighbors is an nb object: list of length = length(id_order),

# each element is an integer vector of neighbor positions in id_order.
  # We simply return it as-is (already positional indices into id_order).
  # Remove any 0-entries (spdep convention for "no neighbors").
  lapply(rook_neighbors, function(nb_idx) {
    nb_idx <- nb_idx[nb_idx != 0L]
    as.integer(nb_idx)
  })
}

# =============================================================================
# STEP 2: Reshape a variable from long cell-year data to a cells x years matrix
# =============================================================================
build_variable_matrix <- function(cell_data, id_order, years, var_name) {
  # Create a matrix: rows = cells (in id_order order), cols = years
  n_cells <- length(id_order)
  n_years <- length(years)

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  colnames(mat) <- as.character(years)

  # Build a fast mapping from cell id to row index in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Map each row of cell_data to (cell_pos, year_col)
  cell_pos <- id_to_pos[as.character(cell_data$id)]
  year_col <- match(cell_data$year, years)

  # Fill the matrix
  valid <- !is.na(cell_pos) & !is.na(year_col)
  mat[cbind(cell_pos[valid], year_col[valid])] <- cell_data[[var_name]][valid]

  mat
}

# =============================================================================
# STEP 3: Compute neighbor max, min, mean for one variable across all years
# =============================================================================
compute_neighbor_stats_optimized <- function(var_matrix, cell_neighbor_lookup) {
  # var_matrix: n_cells x n_years matrix
  # cell_neighbor_lookup: list of length n_cells, each element = integer vector
  #                       of neighbor row indices into var_matrix
  #
  # Returns: list with three matrices (max, min, mean), each n_cells x n_years

  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)

  mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_lookup[[i]]
    if (length(nb) == 0L) next

    if (length(nb) == 1L) {
      # Single neighbor: row vector
      nb_vals <- var_matrix[nb, , drop = FALSE]  # 1 x n_years
      mat_max[i, ]  <- nb_vals[1L, ]
      mat_min[i, ]  <- nb_vals[1L, ]
      mat_mean[i, ] <- nb_vals[1L, ]
    } else {
      # Multiple neighbors: sub-matrix, compute column-wise stats
      nb_vals <- var_matrix[nb, , drop = FALSE]  # length(nb) x n_years

      # Use colMeans, and vectorized pmin/pmax via Reduce or matrixStats
      # For robustness with NAs, use na.rm = TRUE
      mat_max[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      mat_min[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mat_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }

  # Fix Inf/-Inf from max/min on all-NA columns (apply with na.rm on empty)
  mat_max[is.infinite(mat_max)] <- NA_real_
  mat_min[is.infinite(mat_min)] <- NA_real_
  mat_mean[is.nan(mat_mean)]    <- NA_real_

  list(max = mat_max, min = mat_min, mean = mat_mean)
}

# =============================================================================
# STEP 3b: Faster version using matrixStats (if available) — recommended
# =============================================================================
compute_neighbor_stats_fast <- function(var_matrix, cell_neighbor_lookup) {
  require(matrixStats)

  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)

  mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_lookup[[i]]
    if (length(nb) == 0L) next

    if (length(nb) == 1L) {
      row_vals <- var_matrix[nb, ]
      mat_max[i, ]  <- row_vals
      mat_min[i, ]  <- row_vals
      mat_mean[i, ] <- row_vals
    } else {
      nb_vals <- var_matrix[nb, , drop = FALSE]
      mat_max[i, ]  <- colMaxs(nb_vals, na.rm = TRUE)
      mat_min[i, ]  <- colMins(nb_vals, na.rm = TRUE)
      mat_mean[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
    }
  }

  mat_max[is.infinite(mat_max)] <- NA_real_
  mat_min[is.infinite(mat_min)] <- NA_real_
  mat_mean[is.nan(mat_mean)]    <- NA_real_

  list(max = mat_max, min = mat_min, mean = mat_mean)
}

# =============================================================================
# STEP 4: Unpack matrix results back into the long cell_data data.frame
# =============================================================================
unpack_matrix_to_long <- function(cell_data, stat_matrix, id_order, years,
                                  col_name) {
  # stat_matrix: n_cells x n_years, rows aligned to id_order, cols to years
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_pos  <- id_to_pos[as.character(cell_data$id)]
  year_col  <- match(cell_data$year, years)
  valid     <- !is.na(cell_pos) & !is.na(year_col)

  vals <- rep(NA_real_, nrow(cell_data))
  vals[valid] <- stat_matrix[cbind(cell_pos[valid], year_col[valid])]

  cell_data[[col_name]] <- vals
  cell_data
}

# =============================================================================
# STEP 5: Full optimized pipeline (drop-in replacement for the outer loop)
# =============================================================================
run_optimized_neighbor_pipeline <- function(cell_data, id_order,
                                            rook_neighbors_unique) {

  years <- sort(unique(cell_data$year))
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # --- Static topology: built ONCE ---
  message("Building cell-level neighbor lookup (year-invariant)...")
  cell_nb <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)
  compute_fn <- if (use_matrixStats) compute_neighbor_stats_fast else
                                     compute_neighbor_stats_optimized

  # --- Dynamic variables: per-variable matrix operations ---
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing neighbor stats for: %s", var_name))

    # Reshape long -> wide matrix (cells x years)
    var_mat <- build_variable_matrix(cell_data, id_order, years, var_name)

    # Compute neighbor max, min, mean across all years at once
    stats <- compute_fn(var_mat, cell_nb)

    # Unpack back to long format and attach to cell_data
    # Column naming convention preserved from original:
    #   neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
    cell_data <- unpack_matrix_to_long(
      cell_data, stats$max, id_order, years,
      paste0("neighbor_max_", var_name)
    )
    cell_data <- unpack_matrix_to_long(
      cell_data, stats$min, id_order, years,
      paste0("neighbor_min_", var_name)
    )
    cell_data <- unpack_matrix_to_long(
      cell_data, stats$mean, id_order, years,
      paste0("neighbor_mean_", var_name)
    )

    # Free memory
    rm(var_mat, stats)
    gc()
  }

  message("Neighbor feature computation complete.")
  cell_data
}

# =============================================================================
# USAGE (replaces the original outer loop)
# =============================================================================
cell_data <- run_optimized_neighbor_pipeline(
  cell_data, id_order, rook_neighbors_unique
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup entries | ~6.46M (cell × year) | ~344K (cell only) |
| String key operations | ~38M paste + match | 0 |
| R-level `lapply` calls per variable | ~6.46M | ~344K |
| Years processed per cell visit | 1 | 28 (vectorized columns) |
| Estimated wall time | 86+ hours | ~2–4 hours |
| Peak RAM | High (6.46M-element list + strings) | Moderate (344K × 28 matrix) |
| Numerical result | max/min/mean of neighbor values | Identical |
| RF model | Unchanged | Unchanged |

The key insight: **factor the computation into (static topology) × (dynamic values)** instead of materializing the Cartesian product. The neighbor graph is traversed once per cell; the year dimension is handled by columnar matrix operations inside that single traversal.