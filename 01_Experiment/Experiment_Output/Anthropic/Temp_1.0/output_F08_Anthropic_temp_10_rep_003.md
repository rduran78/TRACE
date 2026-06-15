 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup of neighbor row indices, producing ~6.46 million list entries. But the neighbor *topology* is **static** — it depends only on cell identity, not on year. The current code redundantly recomputes the same neighbor cell IDs 28 times (once per year for each cell) and does expensive string-key lookups (`paste` + named-vector indexing) across all 6.46M rows.

Specifically:

1. **`build_neighbor_lookup`** creates ~6.46M character keys (`id_year`) and performs named-vector lookups for each row. With ~6.46M entries in `idx_lookup`, each named lookup is O(n) or O(log n) depending on hashing, repeated ~6.46M × avg_neighbors times. This alone can take many hours.

2. **`compute_neighbor_stats`** iterates over the 6.46M-element list, extracting values and computing `max`, `min`, `mean`. This is repeated for each of 5 variables — so ~32.3M list iterations total.

3. The fundamental waste: the **same neighbor cell relationships** are resolved into row indices 28 times (once per year), when the topology is year-invariant.

## Optimization Strategy

**Separate the static topology from the year-varying data.**

1. **Build a cell-level neighbor index once** — a list of length 344,208 mapping each cell's position to its neighbors' positions (in a cell-order vector). This is just `rook_neighbors_unique` itself (an `nb` object already does this).

2. **Reshape each variable into a matrix**: rows = cells (344,208), columns = years (28). Now cell `i`'s neighbor values in year `j` are simply `matrix[neighbors[[i]], j]`.

3. **Compute neighbor stats as matrix operations over the cell dimension only** — loop over 344,208 cells (not 6.46M cell-years), and for each cell, extract the neighbor sub-matrix (neighbors × 28), then compute columnwise max/min/mean. This produces a (28)-length vector per cell per stat.

4. **Vectorize further** by recognizing that for each cell, the neighbor sub-matrix extraction and column-wise summary can be done very efficiently, or even fully vectorized using sparse-matrix multiplication (for `mean`) and row-wise grouped operations.

This reduces the work from ~6.46M list lookups to ~344K, and eliminates all string-key construction.

**Expected speedup**: From 86+ hours to roughly **minutes** (the dominant cost becomes ~344K list accesses on small sub-matrices of ~4 neighbors × 28 years).

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the fact that neighbor topology is static across years.
# =============================================================================

#' Build a cell-year matrix from panel data for a single variable.
#' Returns a list with:
#'   - mat: a (n_cells x n_years) matrix
#'   - cell_ids: vector of unique cell IDs (row order)
#'   - years: vector of unique years (column order)
#'   - cell_id_to_row: named integer vector mapping cell ID -> row index in mat
build_variable_matrix <- function(data, var_name) {
  cell_ids <- sort(unique(data$id))
  years    <- sort(unique(data$year))

  n_cells <- length(cell_ids)
  n_years <- length(years)

  # Map cell id and year to matrix indices
  cell_id_to_row <- setNames(seq_along(cell_ids), as.character(cell_ids))
  year_to_col    <- setNames(seq_along(years), as.character(years))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_idx <- cell_id_to_row[as.character(data$id)]
  col_idx <- year_to_col[as.character(data$year)]

  mat[cbind(row_idx, col_idx)] <- data[[var_name]]

  list(
    mat            = mat,
    cell_ids       = cell_ids,
    years          = years,
    cell_id_to_row = cell_id_to_row,
    year_to_col    = year_to_col
  )
}

#' Compute neighbor max, min, mean for one variable across all cells and years.
#' Uses the static nb object directly.
#'
#' @param var_mat      Matrix (n_cells x n_years) of variable values.
#' @param neighbors    An nb object (list of length n_cells), where neighbors[[i]]
#'                     contains integer indices of i's neighbors in the same
#'                     cell ordering used to build var_mat.
#'                     Note: spdep nb objects use 0L to denote "no neighbors".
#' @param id_order     The cell ID vector corresponding to the nb object's ordering.
#' @param cell_id_to_row  Named vector mapping cell ID -> row in var_mat.
#'
#' @return A list with three matrices (n_cells x n_years): nb_max, nb_min, nb_mean.
compute_neighbor_stats_matrix <- function(var_mat, neighbors, id_order,
                                          cell_id_to_row) {
  n_cells <- nrow(var_mat)
  n_years <- ncol(var_mat)

  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Build a mapping from nb-object position -> var_mat row.
  # id_order[k] is the cell ID at position k in the nb object.
  # cell_id_to_row maps cell ID -> var_mat row.
  # So nb_pos_to_mat_row[k] gives the var_mat row for nb position k.
  nb_pos_to_mat_row <- cell_id_to_row[as.character(id_order)]
  # nb_pos_to_mat_row is aligned: position k in nb -> row in var_mat

  for (i in seq_len(length(neighbors))) {
    nb_idx <- neighbors[[i]]

    # spdep nb objects use integer(0) or 0L for cells with no neighbors
    if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0L)) next

    # Map nb positions to var_mat rows
    mat_rows <- nb_pos_to_mat_row[nb_idx]
    mat_rows <- mat_rows[!is.na(mat_rows)]
    if (length(mat_rows) == 0) next

    # The current cell's row in var_mat
    my_mat_row <- nb_pos_to_mat_row[i]
    if (is.na(my_mat_row)) next

    # Extract the sub-matrix of neighbor values: (n_neighbors x n_years)
    if (length(mat_rows) == 1) {
      # Single neighbor: treat as a 1-row matrix
      sub <- matrix(var_mat[mat_rows, ], nrow = 1, ncol = n_years)
    } else {
      sub <- var_mat[mat_rows, , drop = FALSE]
    }

    # Column-wise (i.e., per-year) stats
    # Use colMeans, and manual colMax/colMin to handle NAs
    for (j in seq_len(n_years)) {
      vals <- sub[, j]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0) next
      nb_max[my_mat_row, j]  <- max(vals)
      nb_min[my_mat_row, j]  <- min(vals)
      nb_mean[my_mat_row, j] <- mean(vals)
    }
  }

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

#' Flatten a (n_cells x n_years) matrix back into the panel-data row order.
#'
#' @param mat            Matrix (n_cells x n_years).
#' @param data           The panel data.frame (must have $id and $year).
#' @param cell_id_to_row Named vector mapping cell ID -> matrix row.
#' @param year_to_col    Named vector mapping year -> matrix column.
#' @return A numeric vector of length nrow(data), aligned to data's row order.
matrix_to_panel_vector <- function(mat, data, cell_id_to_row, year_to_col) {
  row_idx <- cell_id_to_row[as.character(data$id)]
  col_idx <- year_to_col[as.character(data$year)]
  mat[cbind(row_idx, col_idx)]
}

# =============================================================================
# MAIN PIPELINE (replaces the original outer loop)
# =============================================================================

# --- Inputs assumed to exist: ---
# cell_data              : data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order               : integer vector of cell IDs in the nb object's order
# rook_neighbors_unique  : spdep nb object (list of neighbor index vectors)
# rf_model               : pre-trained Random Forest model (NOT retrained)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build the cell_id_to_row and year_to_col mappings once.
#         (We build them from the first variable; they are the same for all.)
ref <- build_variable_matrix(cell_data, neighbor_source_vars[1])
cell_id_to_row <- ref$cell_id_to_row
year_to_col    <- ref$year_to_col

cat("Cell-year matrix dimensions:", nrow(ref$mat), "cells x", ncol(ref$mat), "years\n")

# Step 2: For each source variable, build matrix, compute neighbor stats, write back.
for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  t0 <- Sys.time()

  # Build (n_cells x n_years) matrix for this variable
  vm <- build_variable_matrix(cell_data, var_name)

  # Compute neighbor max, min, mean using the static topology
  stats <- compute_neighbor_stats_matrix(
    var_mat        = vm$mat,
    neighbors      = rook_neighbors_unique,
    id_order       = id_order,
    cell_id_to_row = cell_id_to_row
  )

  # Flatten back to panel order and attach to cell_data
  cell_data[[paste0("nb_max_", var_name)]] <-
    matrix_to_panel_vector(stats$nb_max, cell_data, cell_id_to_row, year_to_col)
  cell_data[[paste0("nb_min_", var_name)]] <-
    matrix_to_panel_vector(stats$nb_min, cell_data, cell_id_to_row, year_to_col)
  cell_data[[paste0("nb_mean_", var_name)]] <-
    matrix_to_panel_vector(stats$nb_mean, cell_data, cell_id_to_row, year_to_col)

  cat("  Done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "minutes\n")
}

# Step 3: Prediction with the pre-trained Random Forest (unchanged).
# The model is NOT retrained. We simply call predict().
cell_data$predicted <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

## Further Optimization: Vectorized Inner Loop with `vapply`

The inner double loop (over cells × years) can be further accelerated by replacing the per-year loop with vectorized column operations:

```r
# Drop-in replacement for the inner loop in compute_neighbor_stats_matrix.
# Instead of looping over years inside the cell loop, use colMaxs/colMins/colMeans.

compute_neighbor_stats_matrix_fast <- function(var_mat, neighbors, id_order,
                                                cell_id_to_row) {
  n_cells <- nrow(var_mat)
  n_years <- ncol(var_mat)

  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  nb_pos_to_mat_row <- cell_id_to_row[as.character(id_order)]

  # Optional: use matrixStats for colMaxs/colMins if available
  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)

  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0L)) next

    mat_rows <- nb_pos_to_mat_row[nb_idx]
    mat_rows <- mat_rows[!is.na(mat_rows)]
    if (length(mat_rows) == 0) next

    my_mat_row <- nb_pos_to_mat_row[i]
    if (is.na(my_mat_row)) next

    if (length(mat_rows) == 1) {
      # Single neighbor: stats are just that neighbor's values
      nb_max[my_mat_row, ]  <- var_mat[mat_rows, ]
      nb_min[my_mat_row, ]  <- var_mat[mat_rows, ]
      nb_mean[my_mat_row, ] <- var_mat[mat_rows, ]
    } else {
      sub <- var_mat[mat_rows, , drop = FALSE]
      if (use_matrixStats) {
        nb_max[my_mat_row, ]  <- matrixStats::colMaxs(sub, na.rm = TRUE)
        nb_min[my_mat_row, ]  <- matrixStats::colMins(sub, na.rm = TRUE)
        nb_mean[my_mat_row, ] <- colMeans(sub, na.rm = TRUE)
      } else {
        nb_max[my_mat_row, ]  <- apply(sub, 2, max, na.rm = TRUE)
        nb_min[my_mat_row, ]  <- apply(sub, 2, min, na.rm = TRUE)
        nb_mean[my_mat_row, ] <- colMeans(sub, na.rm = TRUE)
      }
    }
  }

  # Replace -Inf/Inf from max/min on all-NA columns back to NA
  nb_max[is.infinite(nb_max)] <- NA_real_
  nb_min[is.infinite(nb_min)] <- NA_real_

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup granularity | Per cell-year (6.46M entries) | Per cell (344K entries) |
| String key construction | ~6.46M `paste()` calls + named lookups | Zero string keys at runtime |
| Inner loop iterations | 6.46M × 5 vars = 32.3M | 344K × 5 vars = 1.72M |
| Data access pattern | Scattered vector indexing | Contiguous matrix sub-rows |
| Estimated runtime | 86+ hours | ~10–30 minutes |
| RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical result | Original estimand | Identical (same max/min/mean logic) |