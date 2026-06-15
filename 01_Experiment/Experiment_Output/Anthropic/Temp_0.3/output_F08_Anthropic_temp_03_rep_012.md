 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination — effectively duplicating the same spatial relationship 28 times per cell.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M paste-based keys (`id_year`) and performs named-vector lookups inside a `lapply` over 6.46M rows. Named vector lookup in R is O(n) in the worst case for each access, and `paste` + string matching over millions of keys is slow.

3. **`compute_neighbor_stats` iterates row-by-row over 6.46M entries** using `lapply`, which is inherently slow in R even when the inner operation is trivial.

4. **The combination** of 6.46M-element `lapply` in `build_neighbor_lookup` and then again in `compute_neighbor_stats` (called 5 times, once per variable) produces the estimated 86+ hour runtime.

**Key insight:** Because the neighbor graph is year-invariant, we can split the problem into:
- A **static spatial lookup** (344K cells → their neighbor cell indices), built once.
- A **year-level matrix operation** where, for each year, we pull the variable values for all cells and compute neighbor max/min/mean using fast vectorized or matrix operations.

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 where each element contains the integer positions of that cell's neighbors within the cell-ID ordering. This is just a cleaned version of `rook_neighbors_unique` (the `nb` object) and costs essentially nothing.

2. **Reshape data so that each year's variable values are accessible as a simple numeric vector indexed by cell position.** We ensure `cell_data` is sorted by `(id, year)` or use a fast index so that for a given year, we can extract a vector of length 344,208 for any variable.

3. **For each variable and each year (28 iterations), vectorize the neighbor aggregation** using the static cell-level neighbor list. This reduces the inner loop from 6.46M iterations to 344K iterations × 28 years = 9.63M, but each iteration is a trivial index-into-vector operation. We can further accelerate with `vapply` or, even better, by constructing a sparse adjacency matrix and using matrix multiplication for the mean, and row-wise operations for max/min.

4. **Sparse matrix approach for mean:** Construct a row-normalized sparse adjacency matrix `W` (344,208 × 344,208). Then `neighbor_mean = W %*% values_vector` is a single sparse matrix-vector multiply per year per variable — extremely fast. For max and min, we use a compiled loop or a grouped operation.

5. **Result:** Instead of ~6.46M `lapply` iterations with string lookups, we get 28 sparse matrix-vector multiplies per variable (for mean) plus fast compiled max/min operations. Total wall-clock time drops from 86+ hours to minutes.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build static spatial structures (done ONCE)
# ==============================================================================

#' Build a sparse row-normalized adjacency matrix from an nb object,
#' plus a raw adjacency list for max/min.
#'
#' @param nb_obj   spdep::nb object (list of integer neighbor indices), length N_cells
#' @param n_cells  number of spatial cells
#' @return list with:
#'   - adj_list: the nb object cleaned (integer neighbor indices per cell)
#'   - W_mean:   sparse Matrix (dgCMatrix), row-normalized for computing means
#'   - W_adj:    sparse binary adjacency Matrix (dgCMatrix) for max/min helpers

build_static_spatial_structures <- function(nb_obj, n_cells) {

  # --- Build sparse adjacency matrix (binary) ---
  # Each entry (i, j) = 1 if cell j is a neighbor of cell i
  from <- rep(seq_len(n_cells), times = lengths(nb_obj))
  to   <- unlist(nb_obj)

  # Remove any 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(to) & to > 0L
  from  <- from[valid]
  to    <- to[valid]

  W_adj <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n_cells, n_cells)
  )

  # --- Row-normalized version for mean computation ---
  row_sums <- rowSums(W_adj)
  row_sums[row_sums == 0] <- NA_real_   # islands get NA
  # Diagonal matrix of 1/row_sums
  D_inv <- Diagonal(x = ifelse(is.na(row_sums), 0, 1 / row_sums))
  W_mean <- D_inv %*% W_adj

  # --- Clean adjacency list (for max/min) ---
  adj_list <- lapply(nb_obj, function(x) {
    x <- as.integer(x)
    x[!is.na(x) & x > 0L]
  })

  list(
    adj_list = adj_list,
    W_mean   = W_mean,
    W_adj    = W_adj,
    n_neighbors = as.integer(row_sums)
  )
}

# ==============================================================================
# STEP 2: Compute neighbor max and min using the adjacency list
# ==============================================================================

#' Fast neighbor max and min for a single numeric vector (one year, one variable).
#' Uses vapply over the static adjacency list.
#'
#' @param values   numeric vector of length n_cells (one value per cell for one year)
#' @param adj_list list of integer vectors (neighbor indices per cell)
#' @return matrix of dimension (n_cells, 2): columns are max, min

neighbor_max_min <- function(values, adj_list) {
  result <- vapply(adj_list, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_))
    nv <- values[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_))
    c(max(nv), min(nv))
  }, numeric(2))
  t(result)  # transpose to n_cells x 2
}

# ==============================================================================
# STEP 3: Compute neighbor mean using sparse matrix multiplication
# ==============================================================================

#' Fast neighbor mean for a single numeric vector (one year, one variable).
#'
#' @param values numeric vector of length n_cells
#' @param W_mean row-normalized sparse adjacency matrix
#' @param n_neighbors integer vector of neighbor counts per cell
#' @return numeric vector of length n_cells

neighbor_mean_sparse <- function(values, W_mean, n_neighbors) {
  # Handle NAs in values: sparse matmul treats them as 0, so we need correction.
  # Strategy: compute sum of non-NA neighbors and count of non-NA neighbors.
  
  is_valid   <- !is.na(values)
  values_0   <- values
  values_0[!is_valid] <- 0  # replace NA with 0 for matmul

  neighbor_sum   <- as.numeric(W_mean %*% values_0) * n_neighbors
  neighbor_count <- as.numeric(W_mean %*% as.numeric(is_valid)) * n_neighbors

  result <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  # Cells with 0 neighbors (islands)
  result[n_neighbors == 0 | is.na(n_neighbors)] <- NA_real_
  result
}

# ==============================================================================
# STEP 4: Main driver — compute all neighbor features for all variables
# ==============================================================================

#' Compute neighbor max, min, mean for all source variables across all years.
#' Adds columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#'
#' @param cell_data     data.frame/data.table with columns: id, year, and all source vars
#' @param id_order      integer vector of cell IDs in the order matching the nb object
#' @param nb_obj        spdep::nb object (rook_neighbors_unique)
#' @param source_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data with new neighbor feature columns appended

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj, source_vars) {

  # Convert to data.table for speed (non-destructive if already data.table)
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))

  message("Building static spatial structures for ", n_cells, " cells...")
  spatial <- build_static_spatial_structures(nb_obj, n_cells)

  # Map cell IDs to their position in id_order (1-based index into adj_list / matrix rows)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Ensure data is keyed for fast subsetting
  setkey(dt, year)

  # Pre-allocate output columns
  for (var_name in source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }

  # Process year by year
  for (yr in years) {
    message("  Processing year ", yr, " ...")

    # Get the subset of rows for this year
    yr_idx <- which(dt$year == yr)
    yr_data <- dt[yr_idx]

    # Map each row's cell ID to its position in the spatial ordering
    cell_positions <- id_to_pos[as.character(yr_data$id)]

    # Build a full-length vector (n_cells) for each variable, indexed by cell position
    for (var_name in source_vars) {
      # Initialize with NA
      full_vec <- rep(NA_real_, n_cells)
      full_vec[cell_positions] <- yr_data[[var_name]]

      # --- Neighbor mean via sparse matrix multiply ---
      n_mean <- neighbor_mean_sparse(full_vec, spatial$W_mean, spatial$n_neighbors)

      # --- Neighbor max and min via adjacency list ---
      mm <- neighbor_max_min(full_vec, spatial$adj_list)

      # Write results back to the data.table rows for this year
      # Map from full cell vector back to the rows present in this year
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")

      set(dt, i = yr_idx, j = col_max,  value = mm[cell_positions, 1])
      set(dt, i = yr_idx, j = col_min,  value = mm[cell_positions, 2])
      set(dt, i = yr_idx, j = col_mean, value = n_mean[cell_positions])
    }
  }

  message("Neighbor feature computation complete.")

  # Return as same class as input
  if (is.data.table(cell_data)) {
    return(dt)
  } else {
    return(as.data.frame(dt))
  }
}

# ==============================================================================
# STEP 5: Usage — drop-in replacement for the original outer loop
# ==============================================================================

# Original code (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data    = cell_data,
  id_order     = id_order,
  nb_obj       = rook_neighbors_unique,
  source_vars  = neighbor_source_vars
)

# The trained Random Forest model is unchanged — proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element list with string key hashing | 344K-element static adjacency list + sparse matrix, built once |
| **Mean computation** | `lapply` over 6.46M rows, per variable | Sparse matrix–vector multiply: 28 years × 5 vars = 140 multiplies on 344K-dim vectors |
| **Max/Min computation** | Embedded in same 6.46M `lapply` | `vapply` over 344K cells × 28 years × 5 vars (compiled inner ops) |
| **Memory** | ~6.46M-element list of integer vectors + string keys | ~1.37M-nonzero sparse matrix (~33 MB) + 344K adjacency list |
| **Estimated runtime** | 86+ hours | ~5–15 minutes on a 16 GB laptop |
| **Numerical result** | — | Identical max, min, mean values per cell-year |
| **RF model** | — | Untouched; no retraining |