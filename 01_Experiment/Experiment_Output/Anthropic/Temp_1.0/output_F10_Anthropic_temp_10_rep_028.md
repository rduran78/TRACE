 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each built via string-pasting and named-vector lookups. The `paste(..., sep="_")` key construction and `idx_lookup[neighbor_keys]` named-vector indexing for every row is O(n) with large constant factors. Named vector lookups in R are hash-based but still slow at this scale.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in a serial `lapply`**, extracting neighbor values, removing NAs, and computing max/min/mean one node-year at a time. This is called 5 times (once per variable), totaling ~32.3 million individual aggregation operations.

3. **The topology is year-invariant but is encoded per-row.** Rook neighbors don't change across years — cell *i*'s spatial neighbors are the same in 1992 as in 2019. Yet the lookup is built at the cell-year level, duplicating identical adjacency information 28 times per cell. This inflates the lookup from ~344K entries to ~6.46M entries.

**Estimated complexity**: Building the lookup is ~O(6.46M × avg_degree × string_ops). Each `compute_neighbor_stats` call is ~O(6.46M × avg_degree). Total: dominated by the 6.46M × 5 × lapply overhead and R-level loops.

## Optimization Strategy

1. **Separate topology from data.** Build a sparse adjacency structure once over the 344,208 cells (not cell-years). Represent it as a CSR (Compressed Sparse Row) structure using integer vectors — no strings, no names, no lists of 6.46M elements.

2. **Vectorize aggregation by year.** For each year, extract cell-level vectors, then use the CSR structure to compute neighbor max/min/mean via vectorized C-level operations. This reduces the problem to 28 iterations × 5 variables × 344K cells, all using vectorized indexing.

3. **Use `data.table` for fast grouped ordering and in-place column assignment.** Avoid copies, avoid `paste`, avoid named lookups.

4. **Use sparse matrix multiplication for `mean`**, and rowwise sparse operations for `max`/`min` via the `Matrix` package. For mean: `A %*% x / degree` where `A` is the binary adjacency matrix. For max/min: operate on CSR pointers directly in vectorized R or via a small C/Rcpp helper.

5. **Preserve numerical equivalence.** The original code computes `max`, `min`, `mean` of non-NA neighbor values. The optimized code must replicate this exactly, including NA handling (nodes with zero non-NA neighbors get NA).

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats.
# Preserves the pre-trained Random Forest model (no retraining).
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build CSR adjacency structure from spdep nb object (ONCE)
#
# rook_neighbors_unique: spdep nb object, length = n_cells = 344208
# id_order: vector of cell IDs of length n_cells, aligning with nb indices
# --------------------------------------------------------------------------

build_csr_from_nb <- function(nb_obj) {
  # nb_obj is a list of integer vectors (neighbor indices, 1-based)
  # Convert to CSR: row_ptr (length n+1), col_idx (length nnz)
  n <- length(nb_obj)
  degrees <- vapply(nb_obj, function(x) {
    # spdep nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  
  nnz <- sum(degrees)
  row_ptr <- c(0L, cumsum(degrees))  # 0-based pointers, length n+1
  col_idx <- integer(nnz)
  
  pos <- 1L
  for (i in seq_len(n)) {
    d <- degrees[i]
    if (d > 0L) {
      col_idx[pos:(pos + d - 1L)] <- nb_obj[[i]]
      pos <- pos + d
    }
  }
  
  list(n = n, nnz = nnz, row_ptr = row_ptr, col_idx = col_idx, degrees = degrees)
}

# --------------------------------------------------------------------------
# STEP 2: Vectorized neighbor aggregation using CSR
#
# For a single variable and single year, compute max/min/mean of neighbor
# values for each cell. Fully vectorized via rep + grouping.
# --------------------------------------------------------------------------

aggregate_neighbors_csr <- function(vals, csr) {
  # vals: numeric vector length n (one value per cell for one year)
  # csr: list with row_ptr, col_idx, degrees, n
  # Returns: n x 3 matrix [max, min, mean], with NA for zero-neighbor or all-NA
  
  n   <- csr$n
  nnz <- csr$nnz
  deg <- csr$degrees
  
  # Preallocate output
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  
  if (nnz == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }
  
  # Expand: for each edge, get the neighbor's value
  neighbor_vals <- vals[csr$col_idx]  # length nnz, vectorized lookup
  
  # Create group IDs: which node does each edge belong to?
  # node i owns edges from row_ptr[i]+1 to row_ptr[i+1]
  # Only for nodes with degree > 0
  has_neighbors <- which(deg > 0L)
  
  if (length(has_neighbors) == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }
  
  group_id <- rep(has_neighbors, times = deg[has_neighbors])  # length nnz
  
  # Use data.table for grouped aggregation (very fast, vectorized C code)
  dt <- data.table(g = group_id, v = neighbor_vals)
  
  # Remove NAs before aggregation (matches original: neighbor_vals[!is.na(...)])
  dt <- dt[!is.na(v)]
  
  if (nrow(dt) == 0L) {
    return(cbind(out_max, out_min, out_mean))
  }
  
  agg <- dt[, .(vmax = max(v), vmin = min(v), vmean = mean(v)), by = g]
  
  out_max[agg$g]  <- agg$vmax
  out_min[agg$g]  <- agg$vmin
  out_mean[agg$g] <- agg$vmean
  
  cbind(out_max, out_min, out_mean)
}

# --------------------------------------------------------------------------
# STEP 3: Main pipeline
# --------------------------------------------------------------------------

run_optimized_neighbor_pipeline <- function(cell_data, id_order, 
                                             rook_neighbors_unique,
                                             rf_model) {
  
  cat("Converting cell_data to data.table...\n")
  setDT(cell_data)
  
  # --- Ensure consistent ordering: cell_data must be indexable by (id, year)
  # Build a mapping from cell ID -> CSR row index (1-based, aligned with nb object)
  n_cells <- length(id_order)
  id_to_csr <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Add CSR index to cell_data
  cell_data[, csr_idx := id_to_csr[as.character(id)]]
  
  # --- Build CSR adjacency (ONCE) ---
  cat("Building CSR adjacency structure...\n")
  csr <- build_csr_from_nb(rook_neighbors_unique)
  cat(sprintf("  Nodes: %d, Edges: %d\n", csr$n, csr$nnz))
  
  # --- Define variables and output column names ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Determine output column names to match original compute_and_add_neighbor_features
  # Original convention (inferred): {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
  make_col_names <- function(var_name) {
    paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }
  
  # Pre-allocate all 15 output columns with NA
  for (var_name in neighbor_source_vars) {
    cols <- make_col_names(var_name)
    for (col in cols) {
      set(cell_data, j = col, value = NA_real_)
    }
  }
  
  # --- Process by year ---
  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d years x %d variables...\n", 
              length(years), length(neighbor_source_vars)))
  
  for (yr in years) {
    cat(sprintf("  Year %d...\n", yr))
    
    # Get row indices for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Get CSR indices for these rows (which cell each row corresponds to)
    yr_csr_idx <- cell_data$csr_idx[yr_rows]
    
    # Build a dense vector for this year: vals_dense[csr_index] = value
    # Some cells may not appear in a given year -> those stay NA
    
    for (var_name in neighbor_source_vars) {
      # Create dense cell-level vector
      vals_dense <- rep(NA_real_, n_cells)
      vals_dense[yr_csr_idx] <- cell_data[[var_name]][yr_rows]
      
      # Compute neighbor aggregation
      stats <- aggregate_neighbors_csr(vals_dense, csr)
      
      # Write results back to the correct rows
      cols <- make_col_names(var_name)
      set(cell_data, i = yr_rows, j = cols[1], value = stats[yr_csr_idx, 1])
      set(cell_data, i = yr_rows, j = cols[2], value = stats[yr_csr_idx, 2])
      set(cell_data, i = yr_rows, j = cols[3], value = stats[yr_csr_idx, 3])
    }
  }
  
  # Clean up helper column
  cell_data[, csr_idx := NULL]
  
  cat("Neighbor feature computation complete.\n")
  
  # --- Apply pre-trained Random Forest (NO retraining) ---
  cat("Generating predictions with pre-trained Random Forest...\n")
  
  # Get the predictor variables the model expects
  pred_vars <- rf_model$forest$independent.variable.names  # ranger
  # Fallback for randomForest package:
  if (is.null(pred_vars) && !is.null(rf_model$xNames)) {
    pred_vars <- rf_model$xNames
  }
  
  # Predict
  if (inherits(rf_model, "ranger")) {
    preds <- predict(rf_model, data = cell_data[, ..pred_vars])
    cell_data[, prediction := preds$predictions]
  } else {
    # randomForest package
    pred_df <- as.data.frame(cell_data[, ..pred_vars])
    cell_data[, prediction := predict(rf_model, newdata = pred_df)]
  }
  
  cat("Pipeline complete.\n")
  return(cell_data)
}

# --------------------------------------------------------------------------
# USAGE:
#
#   result <- run_optimized_neighbor_pipeline(
#     cell_data               = cell_data,
#     id_order                = id_order,
#     rook_neighbors_unique   = rook_neighbors_unique,
#     rf_model                = trained_rf_model
#   )
#
# --------------------------------------------------------------------------
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named vector lookups | One-time CSR build over 344K cells (~1 sec) |
| **Per-variable aggregation** | 6.46M `lapply` iterations with R-level loops | 28 year-slices × vectorized `data.table` grouped aggregation |
| **Memory: lookup** | List of 6.46M integer vectors (~2-4 GB) | 3 integer vectors totaling ~10 MB |
| **String operations** | ~25M+ `paste()` calls | Zero |
| **Total iterations at R level** | ~32.3M `lapply` calls | 140 year×variable iterations (vectorized internals) |
| **Estimated runtime** | 86+ hours | **~5-15 minutes** |
| **Numerical equivalence** | Baseline | Identical (same max/min/mean, same NA handling) |
| **RF model** | Unchanged | Unchanged — predict only, no retraining |

**Key insight**: The topology is static across years. By building the CSR once over 344K cells and iterating over 28 years, we reduce the effective problem size by 28×. Within each year-slice, `data.table`'s grouped aggregation (`[, .(...), by = g]`) uses radix-sort grouping implemented in C, replacing 344K R-level `lapply` calls with a single vectorized operation. The combination yields approximately three orders of magnitude speedup.