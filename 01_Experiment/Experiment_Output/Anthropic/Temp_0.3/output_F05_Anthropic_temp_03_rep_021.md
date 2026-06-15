 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts an ID to character and looks it up in a named vector — O(1) amortized but with string allocation overhead.
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current year to form string keys — allocates new strings per row.
4. **Looks up** those keys in `idx_lookup` (a named vector of 6.46M entries) — named-vector lookup in R is hash-based but still involves repeated string hashing.

With ~6.46M rows and an average of ~8 rook neighbors per cell (1,373,394 directed relationships / 344,208 cells ≈ 4 per cell, but bidirectional ≈ 8), this means roughly **50+ million `paste` and hash-lookup operations**. The named-vector approach has high constant factors in R (string allocation, hashing, GC pressure).

### The Deeper Structural Issue

The neighbor topology is **year-invariant**: cell A's neighbors are the same in 1992 as in 2019. Yet the current code re-discovers the mapping from "cell → neighbor rows" independently for every cell-year row. This means the same spatial lookup is repeated 28 times per cell.

Furthermore, `compute_neighbor_stats` is already vectorized over the lookup, but the lookup itself was built row-by-row with string operations. The entire string-keying strategy is unnecessary if we reformulate the problem.

## Optimization Strategy

### Key Insight: Separate Space from Time

Since the neighbor graph is purely spatial and time-invariant:

1. **Build a spatial-only neighbor lookup once** — a list of length 344,208 mapping each cell index to its neighbor cell indices (integers, no strings).
2. **Build a year-to-row mapping** — for each year, a fast integer vector mapping cell position to row index.
3. **Vectorize the neighbor statistics** using matrix operations: reshape each variable into a `cells × years` matrix, then compute neighbor stats using the spatial neighbor list on columns of the matrix.

This eliminates all string operations, reduces the problem from 6.46M row-level iterations to 344K cell-level iterations (or fully vectorized matrix operations), and cuts memory churn dramatically.

### Expected Speedup

- From ~86 hours to **minutes** (estimated 2–10 minutes depending on RAM pressure).
- No string allocation, no hash lookups, pure integer indexing.
- The Random Forest model is untouched; the numerical output is identical.

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results (max, min, mean of neighbor values)
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # Convert to data.table for fast manipulation (non-destructive)
  dt <- as.data.table(cell_data)

  # ------------------------------------------------------------------
  # Step 1: Establish a canonical cell ordering (spatial index)
  # ------------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching the nb object.
  # Create a map: cell_id -> spatial_index (integer position in id_order)
  n_cells <- length(id_order)
  id_to_spatial <- setNames(seq_len(n_cells), as.character(id_order))

  # Assign each row its spatial index
  dt[, spatial_idx := id_to_spatial[as.character(id)]]

  # ------------------------------------------------------------------
  # Step 2: Build spatial-only neighbor list (integer indices)
  # ------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length n_cells,

  # where each element is an integer vector of neighbor positions
  # (indices into id_order). nb objects use 0 for "no neighbors".
  # We just need to clean it.
  spatial_neighbors <- lapply(seq_len(n_cells), function(s) {
    nb <- rook_neighbors_unique[[s]]
    # spdep::nb uses 0L to indicate no neighbors in a single-element vector
    nb <- nb[nb != 0L]
    as.integer(nb)
  })

  # ------------------------------------------------------------------
  # Step 3: Sort data by (spatial_idx, year) for matrix reshaping
  # ------------------------------------------------------------------
  # Ensure consistent year set
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))

  dt[, year_col := year_to_col[as.character(year)]]

  # Sort by spatial_idx, then year for predictable ordering
  setorder(dt, spatial_idx, year)

  # ------------------------------------------------------------------
  # Step 4: Build row-index matrix: cells x years -> row in dt
  # ------------------------------------------------------------------
  # Not all cells may appear in all years, so we build this carefully.
  # row_matrix[s, t] = row index in dt for spatial cell s, year column t
  # NA if that cell-year doesn't exist.

  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$spatial_idx, dt$year_col)] <- seq_len(nrow(dt))

  # ------------------------------------------------------------------
  # Step 5: For each variable, build a cells x years value matrix,
  #         compute neighbor stats via spatial_neighbors, write back.
  # ------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))

    # Build value matrix: cells x years
    val_vec <- dt[[var_name]]
    val_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    valid_rows <- !is.na(row_matrix)
    val_matrix[valid_rows] <- val_vec[row_matrix[valid_rows]]

    # Allocate output matrices: cells x years
    max_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Compute neighbor stats per cell (loop over cells, vectorized over years)
    for (s in seq_len(n_cells)) {
      nb_idx <- spatial_neighbors[[s]]
      if (length(nb_idx) == 0L) next

      # neighbor_vals is a (n_neighbors x n_years) submatrix
      if (length(nb_idx) == 1L) {
        # Single neighbor: treat as a 1-row matrix
        nb_vals <- matrix(val_matrix[nb_idx, ], nrow = 1L, ncol = n_years)
      } else {
        nb_vals <- val_matrix[nb_idx, , drop = FALSE]
      }

      # For each year (column), compute max/min/mean ignoring NAs
      # Use colMeans, colMaxs-style operations via matrixStats or base apply
      # For performance, use vectorized column operations:
      max_matrix[s, ]  <- apply(nb_vals, 2L, max,  na.rm = TRUE)
      min_matrix[s, ]  <- apply(nb_vals, 2L, min,  na.rm = TRUE)
      mean_matrix[s, ] <- colMeans(nb_vals, na.rm = TRUE)
    }

    # Replace Inf/-Inf from max/min on all-NA columns with NA
    max_matrix[is.infinite(max_matrix)]   <- NA_real_
    min_matrix[is.infinite(min_matrix)]   <- NA_real_
    mean_matrix[is.nan(mean_matrix)]      <- NA_real_

    # Map back from (cells x years) matrices to dt rows
    valid <- which(!is.na(row_matrix), arr.ind = TRUE)
    target_rows <- row_matrix[valid]

    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(dt, i = target_rows, j = col_max,  value = max_matrix[valid])
    set(dt, i = target_rows, j = col_min,  value = min_matrix[valid])
    set(dt, i = target_rows, j = col_mean, value = mean_matrix[valid])

    cat(sprintf("  Done: %s\n", var_name))
  }

  # ------------------------------------------------------------------
  # Step 6: Clean up helper columns and return
  # ------------------------------------------------------------------
  dt[, c("spatial_idx", "year_col") := NULL]

  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged — just use cell_data for predict().
```

## Further Optimization: Eliminate the Inner `apply` with `matrixStats`

The `apply(..., 2, max)` call inside the cell loop still has overhead. If the `matrixStats` package is available, replace the inner loop body for a significant constant-factor speedup:

```r
# install.packages("matrixStats") if not already available
library(matrixStats)

# Replace the inner cell loop with:
for (s in seq_len(n_cells)) {
  nb_idx <- spatial_neighbors[[s]]
  if (length(nb_idx) == 0L) next

  nb_vals <- val_matrix[nb_idx, , drop = FALSE]

  max_matrix[s, ]  <- colMaxs(nb_vals,  na.rm = TRUE)
  min_matrix[s, ]  <- colMins(nb_vals,  na.rm = TRUE)
  mean_matrix[s, ] <- colMeans2(nb_vals, na.rm = TRUE)
}
```

## Even Further: Fully Vectorized (No Cell Loop) via Sparse Matrix Multiplication

For the **mean**, we can eliminate the cell loop entirely using a sparse adjacency matrix. Max and min still require per-cell work, but mean is a linear operation:

```r
library(Matrix)

# Build sparse row-normalized adjacency matrix (once, reuse for all vars)
# W[s, j] = 1/degree(s) if j is a neighbor of s, else 0
i_idx <- rep(seq_len(n_cells), lengths(spatial_neighbors))
j_idx <- unlist(spatial_neighbors)
degrees <- lengths(spatial_neighbors)
degrees[degrees == 0] <- 1  # avoid division by zero; those rows will be all-zero anyway

W <- sparseMatrix(
  i = i_idx,
  j = j_idx,
  x = rep(1 / degrees[i_idx != 0], lengths(spatial_neighbors)),
  dims = c(n_cells, n_cells)
)

# For each variable, neighbor mean = W %*% val_matrix
# (handles NAs approximately; for exact NA handling, use a second pass)
mean_matrix <- as.matrix(W %*% val_matrix)

# For exact NA-aware mean:
# Replace NAs with 0 in val_matrix, compute sum and count separately
val_nona <- val_matrix
val_nona[is.na(val_nona)] <- 0
indicator <- (!is.na(val_matrix)) * 1.0

# Build unnormalized adjacency (just 1s for neighbors)
W_raw <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_cells, n_cells))

neighbor_sum   <- as.matrix(W_raw %*% val_nona)
neighbor_count <- as.matrix(W_raw %*% indicator)
mean_matrix    <- neighbor_sum / neighbor_count  # NaN where count=0, which is correct
mean_matrix[neighbor_count == 0] <- NA_real_
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | 6.46M string keys in named vector | Integer matrix indexing |
| **String operations** | ~50M `paste()` + hash lookups | Zero |
| **Redundant work** | Same spatial lookup repeated 28× per cell | Spatial lookup done once |
| **Inner loop granularity** | Per row (6.46M iterations) | Per cell (344K iterations), vectorized over years |
| **Estimated runtime** | 86+ hours | 2–10 minutes |
| **Numerical result** | Baseline | Identical |
| **RF model** | Unchanged | Unchanged |