 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats this identical structural work 28 times per cell. This results in:

1. **~6.46 million list elements** instead of ~344,208 (one per cell).
2. **String concatenation and hash-table lookups** (`paste`, named-vector indexing) on ~6.46M × avg_neighbors scale — extremely slow in R.
3. **Redundant recomputation**: the neighbor graph is static, but the lookup is rebuilt as if it were year-varying.
4. **`compute_neighbor_stats`** then iterates over 6.46M list entries per variable — 5 variables × 6.46M = ~32.3M R-level `lapply` iterations, each doing subsetting and summary stats.

The 86+ hour estimate is entirely explained by this O(cells × years × neighbors) string-key approach applied at the R interpreter level.

## Optimization Strategy

**Separate topology (static) from data (year-varying):**

1. **Build the neighbor lookup once, at the cell level only** (~344K entries). Map each cell to its position in `id_order`, and store neighbor *positions* (integer indices into `id_order`). This is a one-time O(cells × avg_neighbors) operation with no string manipulation.

2. **Reshape each variable into a matrix**: rows = cells (in `id_order` order), columns = years. This gives O(1) column-vector access to all cells' values for a given year.

3. **Vectorized neighbor stats per year**: For each year-column, use the static cell-level neighbor list to compute max/min/mean of neighbor values. Critically, we can do this with a single `vapply` over ~344K cells (not 6.46M rows), repeated for 28 years — a ~18.7× reduction in iterations. Each iteration is a simple integer-index subset of a numeric vector (no string ops).

4. **Flatten back** to the original cell-year row order and attach columns.

This reduces the dominant cost from ~6.46M × 5 slow string-based lookups to ~344K × 28 × 5 fast integer-vector subsets, a roughly **500–1000× speedup**, bringing runtime to minutes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from year-varying data
# =============================================================================

#' Build a cell-level neighbor index lookup (static, computed once).
#' 
#' @param id_order Integer vector of cell IDs in the order matching rook_neighbors_unique.
#' @param neighbors An spdep::nb object (list of integer index vectors into id_order).
#' @return A list of length length(id_order). Each element is an integer vector
#'         of positions (indices into id_order) of that cell's neighbors.
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is already an nb object: neighbors[[i]] gives integer indices

# into id_order for the neighbors of id_order[i].
  # We just need to clean it (remove 0L entries that spdep uses for "no neighbors").
  n <- length(id_order)
  lookup <- vector("list", n)
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb <- nb[nb != 0L]
    lookup[[i]] <- nb
  }
  lookup
}

#' Build a mapping from (cell_id, year) in cell_data to (cell_position, year_index).
#' Returns the year vector (sorted unique years) and a matrix of variable values
#' with rows = cells (in id_order order) and columns = years.
#'
#' @param cell_data Data frame with columns id, year, and variable columns.
#' @param id_order Integer vector of cell IDs.
#' @param var_name Character: name of the variable to extract.
#' @return A list with:
#'   - years: sorted unique year vector
#'   - mat: numeric matrix [length(id_order) x length(years)]
build_variable_matrix <- function(cell_data, id_order, var_name) {
  years <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Create mapping from cell id to row-position in matrix
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  # If IDs are not contiguous or max is very large, use a hash instead:
  # But for 344K cells this is fine (max ~344K integers = ~1.4 MB)
  
  # Create mapping from year to column-position
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # Allocate matrix

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Fill matrix
  row_pos <- id_to_pos[cell_data$id]
  col_pos <- year_to_col[as.character(cell_data$year)]
  mat[cbind(row_pos, col_pos)] <- cell_data[[var_name]]
  
  list(years = years, mat = mat)
}

#' Compute neighbor max, min, mean for one variable across all cell-years.
#'
#' @param cell_data Data frame (original, with id and year columns).
#' @param id_order Integer vector of cell IDs.
#' @param cell_neighbor_lookup List from build_cell_neighbor_lookup.
#' @param var_name Character: source variable name.
#' @return A data frame with three columns: {var}_neighbor_max, {var}_neighbor_min,
#'         {var}_neighbor_mean, in the same row order as cell_data.
compute_neighbor_stats_optimized <- function(cell_data, id_order, 
                                              cell_neighbor_lookup, var_name) {
  # Step 1: Reshape variable into cell × year matrix
  var_info <- build_variable_matrix(cell_data, id_order, var_name)
  mat      <- var_info$mat      # [n_cells x n_years]
  years    <- var_info$years
  n_cells  <- length(id_order)
  n_years  <- length(years)
  
  # Step 2: Allocate output matrices (same shape)
  out_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  out_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  out_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Step 3: For each year, compute neighbor stats using the static topology
  for (j in seq_len(n_years)) {
    vals_j <- mat[, j]  # all cell values for this year
    
    # vapply over cells — ~344K iterations per year
    stats_j <- vapply(cell_neighbor_lookup, function(nb_idx) {
      if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nv <- vals_j[nb_idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nv), min(nv), mean(nv))
    }, numeric(3))
    # stats_j is [3 x n_cells]
    
    out_max[, j]  <- stats_j[1L, ]
    out_min[, j]  <- stats_j[2L, ]
    out_mean[, j] <- stats_j[3L, ]
  }
  
  # Step 4: Map results back to cell_data row order
  # Reconstruct the same positional mappings used in build_variable_matrix
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  row_pos <- id_to_pos[cell_data$id]
  col_pos <- year_to_col[as.character(cell_data$year)]
  linear_idx <- (col_pos - 1L) * n_cells + row_pos
  
  data.frame(
    nb_max  = out_max[linear_idx],
    nb_min  = out_min[linear_idx],
    nb_mean = out_mean[linear_idx]
  )
}

# =============================================================================
# MAIN EXECUTION — replaces the original outer loop
# =============================================================================

# Build the static cell-level neighbor lookup ONCE
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  
  stats_df <- compute_neighbor_stats_optimized(
    cell_data, id_order, cell_neighbor_lookup, var_name
  )
  
  # Attach columns with original naming convention
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats_df$nb_max
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats_df$nb_min
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats_df$nb_mean
}

# cell_data now has all 15 neighbor feature columns (5 vars × 3 stats)
# and is ready for predict(trained_rf_model, newdata = cell_data)
```

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` computed on the same neighbor value sets — no approximation. |
| **Trained RF model preserved** | No retraining; code only prepares predictor columns for `predict()`. |
| **Column names preserved** | `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` — same as original `compute_and_add_neighbor_features` would produce. (Adjust the `paste0` suffix if your original convention differs.) |
| **Memory** | Peak: three `344K × 28` float matrices (~230 MB) + the cell-level lookup (~50 MB) + `cell_data` itself. Well within 16 GB. |
| **Expected runtime** | ~344K cells × 28 years × 5 vars = ~48.2M `vapply` iterations of trivial integer-subset work. Estimated **5–15 minutes** on a standard laptop (vs. 86+ hours). |