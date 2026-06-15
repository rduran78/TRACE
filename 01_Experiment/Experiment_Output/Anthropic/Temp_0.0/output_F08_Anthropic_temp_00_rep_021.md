 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. This means:

1. **Redundant lookup construction**: The neighbor graph is spatial and static — cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds string-keyed lookups for every cell-year combination, creating ~6.46M entries instead of ~344K.

2. **String-key hashing is expensive at scale**: `paste(id, year, sep="_")` and named-vector lookups over 6.46M keys are O(n) in memory and slow in R.

3. **`compute_neighbor_stats` iterates over 6.46M list entries**: Each call to `lapply` over the full row-level lookup, repeated for 5 variables, means ~32.3M list-element iterations total.

4. **The fix is conceptually simple**: Build the neighbor lookup **once over the 344K cells** (not cell-years). Then, for each year, subset the relevant rows, pull neighbor values using the cell-level index, and compute stats. This reduces the lookup from 6.46M entries to 344K entries and makes the per-year computation a fast matrix operation.

## Optimization Strategy

1. **Build a cell-level neighbor index once** (`cell_neighbor_idx`): A list of length 344,208 where each element contains the integer positions of that cell's neighbors within the ordered cell vector. This is year-independent.

2. **Reshape each variable into a matrix**: rows = cells (in `id_order` order), columns = years. This allows vectorized column-slice access.

3. **Compute neighbor stats per year via vectorized operations**: For each year (column), use the static neighbor index to gather neighbor values and compute max/min/mean. This can be done with `vapply` over cells within each year — or even more efficiently with a sparse-matrix approach.

4. **Flatten results back into the original data.frame column order** to preserve downstream compatibility with the pre-trained Random Forest model.

**Expected speedup**: The lookup shrinks by 28×. The per-variable computation becomes a loop over 28 years × 344K cells with simple integer indexing (no string hashing). Estimated runtime drops from 86+ hours to minutes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the fact that neighbor topology is static across years.
# =============================================================================

#' Build a CELL-level neighbor lookup (year-independent).
#' Returns a list of length n_cells. Each element is an integer vector of
#' neighbor positions in id_order.
#'
#' @param id_order   Integer vector of cell IDs in the order matching the nb object.
#' @param neighbors  An spdep::nb object (list of integer index vectors).
#' @return A named list keyed by cell ID (as character), values are integer
#'         vectors of positions in id_order.
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  n <- length(id_order)
  stopifnot(length(neighbors) == n)
  # neighbors[[i]] already contains integer indices into id_order

  # We just need to ensure they are clean integer vectors.
  lookup <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep::nb uses 0L to denote "no neighbors"; filter those out
    nb_i <- nb_i[nb_i > 0L]
    lookup[[i]] <- as.integer(nb_i)
  }
  names(lookup) <- as.character(id_order)
  lookup
}

#' Compute neighbor max, min, mean for one variable across all cell-years.
#'
#' @param cell_data           data.frame with columns: id, year, and the target variable.
#' @param var_name            Character: name of the variable column.
#' @param cell_neighbor_lookup List from build_cell_neighbor_lookup().
#' @param id_order            Integer vector of cell IDs in canonical order.
#' @param years               Sorted integer vector of unique years.
#' @return A data.frame with three columns: <var>_neighbor_max, _min, _mean,
#'         in the same row order as cell_data.
compute_neighbor_stats_optimized <- function(cell_data,
                                              var_name,
                                              cell_neighbor_lookup,
                                              id_order,
                                              years) {
  n_cells <- length(id_order)
  n_years <- length(years)
  n_rows  <- nrow(cell_data)

  # --- Step 1: Build a cell-position lookup for fast mapping ----------------

  # Map cell id -> position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # --- Step 2: Reshape variable into a matrix [n_cells x n_years] -----------
  # We need to map each row of cell_data to (cell_pos, year_col).
  cell_pos_vec <- id_to_pos[as.character(cell_data$id)]
  year_to_col  <- setNames(seq_along(years), as.character(years))
  year_col_vec <- year_to_col[as.character(cell_data$year)]

  val_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- cell_data[[var_name]]
  # Fill the matrix
  idx_linear <- (year_col_vec - 1L) * n_cells + cell_pos_vec
  val_matrix[idx_linear] <- vals

  # --- Step 3: Compute neighbor stats per cell (vectorized over years) ------
  # Result matrices: [n_cells x n_years]
  max_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb_idx <- cell_neighbor_lookup[[i]]
    if (length(nb_idx) == 0L) next
    # nb_vals is a matrix: [length(nb_idx) x n_years]
    # Each row is one neighbor, each column is one year.
    nb_vals <- val_matrix[nb_idx, , drop = FALSE]
    # Compute column-wise stats (across neighbors, for each year)
    # Using colMeans / apply for max/min — but we can be smarter:
    if (length(nb_idx) == 1L) {
      # Single neighbor: max = min = mean = that value
      max_matrix[i, ]  <- nb_vals[1L, ]
      min_matrix[i, ]  <- nb_vals[1L, ]
      mean_matrix[i, ] <- nb_vals[1L, ]
    } else {
      # suppressWarnings to handle all-NA columns gracefully
      max_matrix[i, ]  <- suppressWarnings(apply(nb_vals, 2L, max,  na.rm = TRUE))
      min_matrix[i, ]  <- suppressWarnings(apply(nb_vals, 2L, min,  na.rm = TRUE))
      mean_matrix[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }

  # Fix Inf/-Inf from max/min on all-NA slices
  max_matrix[is.infinite(max_matrix)]  <- NA_real_
  min_matrix[is.infinite(min_matrix)]  <- NA_real_
  mean_matrix[is.nan(mean_matrix)]     <- NA_real_

  # --- Step 4: Flatten back to cell_data row order --------------------------
  out_max  <- max_matrix[idx_linear]
  out_min  <- min_matrix[idx_linear]
  out_mean <- mean_matrix[idx_linear]

  result <- data.frame(out_max, out_min, out_mean)
  colnames(result) <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  result
}


# =============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# =============================================================================

# --- 1. Build the static cell-level neighbor lookup ONCE ---
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# --- 2. Identify canonical orderings ---
years <- sort(unique(cell_data$year))

# --- 3. Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats_df <- compute_neighbor_stats_optimized(
    cell_data            = cell_data,
    var_name             = var_name,
    cell_neighbor_lookup = cell_neighbor_lookup,
    id_order             = id_order,
    years                = years
  )
  # Attach the three new columns to cell_data
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats_df[[1L]]
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats_df[[2L]]
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats_df[[3L]]
}

# --- 4. Predict with the pre-trained Random Forest (UNCHANGED) ---
# The model object and predict() call remain exactly as before.
# cell_data now has the same columns with the same numerical values.
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

## Why This Preserves the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `cell_neighbor_lookup` uses the same `rook_neighbors_unique` nb object and `id_order`. The integer indices in the nb object directly index into `id_order`, so the neighbor sets are identical. |
| **Same aggregation functions** | `max`, `min`, `mean` with `na.rm = TRUE` are applied to the same neighbor value vectors. The only difference is vectorization across years via matrix columns. |
| **Same NA handling** | All-NA neighbor sets produce `NA` (via `Inf`→`NA` correction and `NaN`→`NA`), matching the original `return(c(NA, NA, NA))` logic. |
| **Same column names** | Output columns are named `<var>_neighbor_max`, `<var>_neighbor_min`, `<var>_neighbor_mean` — matching the original `compute_and_add_neighbor_features` output. |
| **RF model untouched** | No retraining. The model object is used as-is for `predict()`. |

## Performance Summary

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup entries | ~6.46M (cell-year) | ~344K (cell only) |
| String hashing operations | ~6.46M × 2 (build + query) | 0 at query time |
| Core loop iterations | ~6.46M per variable | ~344K per variable (matrix ops over 28 years) |
| Estimated wall time | 86+ hours | ~10–30 minutes |
| Peak RAM | Comparable | ~344K × 28 × 8 bytes ≈ 77 MB per variable matrix (fits easily in 16 GB) |