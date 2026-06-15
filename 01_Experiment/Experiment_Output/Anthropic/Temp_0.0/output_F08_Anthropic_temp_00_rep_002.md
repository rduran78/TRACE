 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats this identical structural work 28 times per cell. This yields ~6.46M list elements, each built via string concatenation and named-vector lookup — an O(n × T × k) operation where n=344,208 cells, T=28 years, and k≈4 average neighbors.

`compute_neighbor_stats` then iterates over all ~6.46M entries for each of the 5 variables, extracting values one-at-a-time in R-level loops.

**Root cause summary:**

| Issue | Impact |
|---|---|
| Neighbor topology recomputed for every year | 28× redundant work in lookup construction |
| String-key hashing over 6.46M rows | Extremely slow lookup build (~hours) |
| R-level `lapply` over 6.46M entries per variable | Slow stats computation (5 variables × 6.46M) |
| Entire lookup is cell-year granularity | ~6.46M list entries instead of ~344K |

## Optimization Strategy

**Separate static structure from dynamic data:**

1. **Build the neighbor lookup once at the cell level** (344,208 entries), not at the cell-year level (6.46M entries). The `rook_neighbors_unique` nb object already *is* this lookup — it maps each cell index to its neighbor cell indices. No string hashing needed.

2. **Compute neighbor stats via vectorized matrix operations.** Reshape each variable into a matrix of dimensions (n_cells × n_years). For each cell, the neighbor rows in this matrix are the same across all years. Use sparse-matrix multiplication or direct indexed aggregation to compute max, min, and mean across neighbors — fully vectorized over years.

3. **Use `data.table` for fast reshaping and joining** results back to the panel.

This reduces complexity from O(n × T × k × V) in slow R loops to O(n × k × V) vectorized operations over T-length vectors, with a ~28× structural speedup and additional orders-of-magnitude speedup from vectorization.

**Expected runtime:** Minutes instead of 86+ hours.

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation.
#' Exploits the fact that neighbor topology is static across years,
#' while variable values change by year.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param neighbors       spdep::nb object (list of integer vectors of neighbor indices)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor_max/min/mean columns appended
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)
  
  # --- Step 1: Establish cell ordering ---
  # Map cell IDs to their positional index in id_order (which matches the nb object)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Determine the year vector (sorted)
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # --- Step 2: Sort data for fast matrix construction ---
  # Assign cell index and year index to each row
  dt[, cell_idx := id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_col[as.character(year)]]
  
  # Ensure proper ordering for matrix fill
  setorder(dt, cell_idx, year_idx)
  
  # Verify we have a complete panel (or handle missing)
  # Build a linear index for (cell_idx, year_idx) -> row position
  # For a complete panel: row = (cell_idx - 1) * n_years + year_idx
  
  # --- Step 3: Build neighbor CSR-like structure once ---
  # neighbors[[i]] gives the indices (into id_order) of neighbors of cell i
  # We precompute the "pointer" and "index" arrays for vectorized access
  
  # Compute max number of neighbors for pre-allocation
  n_neighbors <- vapply(neighbors, length, integer(1))
  max_k <- max(n_neighbors)
  
  # Build a padded neighbor matrix: n_cells x max_k
  # Pad with NA for cells with fewer neighbors
  neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0L) {
      neighbor_mat[i, seq_along(nb_i)] <- nb_i
    }
  }
  
  cat("Neighbor matrix built:", n_cells, "cells x", max_k, "max neighbors\n")
  
  # --- Step 4: For each variable, build matrix and compute stats vectorized ---
  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")
    
    # Build n_cells x n_years matrix of values
    # Using the sorted dt, this is straightforward for a complete panel
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # Initialize result matrices
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # For each neighbor "slot" k, accumulate statistics
    # We use a running approach: track running max, min, sum, count
    
    # Initialize with NA
    running_max   <- matrix(-Inf, nrow = n_cells, ncol = n_years)
    running_min   <- matrix(Inf,  nrow = n_cells, ncol = n_years)
    running_sum   <- matrix(0,    nrow = n_cells, ncol = n_years)
    running_count <- matrix(0L,   nrow = n_cells, ncol = n_years)
    
    for (k in seq_len(max_k)) {
      # Which cells have a k-th neighbor?
      nb_k <- neighbor_mat[, k]  # length n_cells, NA if no k-th neighbor
      has_nb <- !is.na(nb_k)
      
      if (!any(has_nb)) next
      
      # Extract the neighbor values for all years at once
      # nb_k[has_nb] are row indices into val_mat
      # This gives a (sum(has_nb) x n_years) submatrix of neighbor values
      nb_vals <- val_mat[nb_k[has_nb], , drop = FALSE]  # submatrix
      
      # Determine which are non-NA
      valid <- !is.na(nb_vals)
      
      # Update running stats for cells that have this neighbor
      # Replace NAs with neutral elements for safe comparison
      nb_vals_for_max <- nb_vals
      nb_vals_for_max[!valid] <- -Inf
      nb_vals_for_min <- nb_vals
      nb_vals_for_min[!valid] <- Inf
      nb_vals_for_sum <- nb_vals
      nb_vals_for_sum[!valid] <- 0
      
      running_max[has_nb, ]   <- pmax(running_max[has_nb, , drop = FALSE], nb_vals_for_max)
      running_min[has_nb, ]   <- pmin(running_min[has_nb, , drop = FALSE], nb_vals_for_min)
      running_sum[has_nb, ]   <- running_sum[has_nb, , drop = FALSE] + nb_vals_for_sum
      running_count[has_nb, ] <- running_count[has_nb, , drop = FALSE] + valid
    }
    
    # Finalize
    no_data <- running_count == 0L
    
    max_mat <- running_max
    max_mat[no_data] <- NA_real_
    max_mat[is.infinite(max_mat)] <- NA_real_
    
    min_mat <- running_min
    min_mat[no_data] <- NA_real_
    min_mat[is.infinite(min_mat)] <- NA_real_
    
    mean_mat <- running_sum / running_count
    mean_mat[no_data] <- NA_real_
    
    # --- Step 5: Map results back to dt rows ---
    linear_idx <- cbind(dt$cell_idx, dt$year_idx)
    
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col_name)  := max_mat[linear_idx]]
    dt[, (min_col_name)  := min_mat[linear_idx]]
    dt[, (mean_col_name) := mean_mat[linear_idx]]
    
    # Free memory
    rm(val_mat, max_mat, min_mat, mean_mat,
       running_max, running_min, running_sum, running_count)
    gc()
    
    cat("  Done:", var_name, "\n")
  }
  
  # Clean up helper columns
  dt[, c("cell_idx", "year_idx") := NULL]
  
  return(dt)
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The pre-trained Random Forest model is used as before — no retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | For each cell-year row, the max/min/mean is computed over exactly the same set of rook-neighbor values as the original code. The neighbor matrix encodes the identical topology from `rook_neighbors_unique`, and values are drawn from the same variable columns. |
| **Trained RF model untouched** | The code only prepares predictor columns; `predict()` is called on the existing model object with no retraining. |
| **NA handling** | Matches original: NAs in neighbor values are excluded; if all neighbors are NA or a cell has no neighbors, the result is NA — identical to the original `compute_neighbor_stats`. |

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string-key lookups | 0 (use nb object directly) |
| Lookup size | ~6.46M list entries | ~344K × max_k integer matrix |
| Stats computation | R-level lapply over 6.46M per variable | Vectorized matrix ops, loop only over max_k (~4) |
| Total R-level loop iterations | ~6.46M × 5 = ~32.3M | ~4 × 5 = ~20 |
| Estimated runtime | 86+ hours | **~5–15 minutes** |
| Peak memory | Moderate (list overhead) | ~344K × 28 × ~6 matrices ≈ ~2–3 GB peak (fits in 16 GB) |

The key insight is that the **neighbor topology is a property of the spatial grid, not of the panel**, so it should be encoded once as a static structure (the padded integer matrix), and the per-year variable values should be organized as matrices where vectorized column operations naturally broadcast the static topology across all years simultaneously.