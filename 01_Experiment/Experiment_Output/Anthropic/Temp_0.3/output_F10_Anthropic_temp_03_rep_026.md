 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups. The `paste(id, year)` keying strategy is O(n) per row with expensive string operations, and the resulting list of 6.46M integer vectors consumes enormous memory and time.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in R-level `lapply`**, performing per-element subsetting, NA removal, and summary statistics. This is inherently slow in interpreted R.

3. **The neighbor topology is year-invariant** (rook contiguity doesn't change across years), but the lookup is rebuilt monolithically across all cell-years, entangling spatial structure with temporal indexing. This prevents vectorized year-sliced computation.

**Key insight:** The rook-neighbor graph is a fixed sparse adjacency matrix over 344,208 cells. For each year, the "neighborhood aggregation" is simply a sparse matrix–vector product (for mean) or sparse row-wise extrema (for max/min). This can be computed in seconds using the `Matrix` package, completely eliminating the per-row R-level loops.

## Optimization Strategy

1. **Build a sparse adjacency matrix `A` (344,208 × 344,208) once** from the `nb` object. Row-normalize a copy for computing means. This is the graph topology, reused for all years and all variables.

2. **For each year, extract the column vector `x` for each variable**, then compute:
   - `neighbor_max`: row-wise max over non-zero entries of `A` with values replaced by `x[j]`
   - `neighbor_min`: row-wise min (same approach)
   - `neighbor_mean`: sparse matrix–vector multiply `A_rownorm %*% x`

3. **Use `Matrix` sparse operations and `data.table` for fast indexing/joining**, eliminating all `paste`-key lookups and R-level `lapply` loops.

4. **Process year-by-year** to keep memory bounded (one 344K-length vector at a time rather than 6.46M).

**Expected speedup:** From 86+ hours to ~5–15 minutes on the same laptop.

## Working R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats output
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build sparse adjacency matrix from nb object (ONCE) ----

build_adjacency_matrix <- function(nb_obj) {
  # nb_obj: list of length n_cells, each element is integer vector of neighbor indices
  n <- length(nb_obj)
  
  # Build COO triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)
  
  # Binary adjacency matrix: A[i,j] = 1 means j is a rook neighbor of i
  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(n, n),
    repr = "C"   # CSC -> we'll convert to CSR via dgRMatrix or use column ops
  )
  
  A
}

# ---- Step 2: Row-wise sparse max, min, mean ----
# For mean: use (A %*% x) / (A %*% ones) = sum of neighbor vals / degree
# For max/min: we need row-wise extrema over {x[j] : A[i,j]=1}

# Efficient row-wise sparse extrema using the CSC structure
# We convert to dgTMatrix (triplet) for direct indexing, then aggregate with data.table

sparse_neighbor_stats <- function(A, x) {
  # A: dgCMatrix (n x n), binary adjacency
  # x: numeric vector length n (variable values for one year)
  # Returns: n x 3 matrix [max, min, mean], NA where no neighbors or all neighbor vals NA
  
  n <- nrow(A)
  
  # Extract triplet form
  At <- as(A, "TsparseMatrix")  # 0-indexed i, j
  row_i <- At@i + 1L   # 1-indexed row
  col_j <- At@j + 1L   # 1-indexed col (the neighbor)
  
  # Get neighbor values
  neighbor_vals <- x[col_j]
  
  # Use data.table for grouped aggregation (extremely fast)
  dt <- data.table(row = row_i, val = neighbor_vals)
  
  # Remove NA neighbor values
  dt <- dt[!is.na(val)]
  
  # Aggregate
  agg <- dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = row]
  
  # Map back to full vector
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("max", "min", "mean")
  result[agg$row, 1L] <- agg$nb_max
  result[agg$row, 2L] <- agg$nb_min
  result[agg$row, 3L] <- agg$nb_mean
  
  result
}

# ---- Step 3: Main pipeline ----

run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data: data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of neighbor index vectors)
  
  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, length(years), nrow(cell_data)))
  
  # --- Build adjacency matrix ONCE ---
  cat("Building sparse adjacency matrix...\n")
  t0 <- proc.time()
  A <- build_adjacency_matrix(rook_neighbors_unique)
  cat(sprintf("  Adjacency matrix: %d x %d, %d non-zeros (%.1f sec)\n",
              nrow(A), ncol(A), nnz(A), (proc.time() - t0)[3]))
  
  # Pre-convert to triplet form ONCE (avoid repeated conversion)
  cat("Converting to triplet form...\n")
  At <- as(A, "TsparseMatrix")
  edge_row <- At@i + 1L
  edge_col <- At@j + 1L
  rm(At)
  
  # --- Build cell ID -> matrix index mapping ---
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Create row key for fast year-sliced access ---
  # Map each cell_data row to its matrix index
  cell_data[, .mat_idx := id_to_idx[as.character(id)]]
  
  # Ensure ordering for fast extraction
  setkey(cell_data, year, .mat_idx)
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  # --- Process year by year ---
  cat("Computing neighbor statistics...\n")
  t_total <- proc.time()
  
  for (yr in years) {
    t_yr <- proc.time()
    
    # Extract rows for this year
    yr_rows <- cell_data[.(yr)]  # keyed lookup
    yr_idx  <- yr_rows$.mat_idx
    
    # For each variable, build the full-length vector, compute stats, assign back
    for (var_name in neighbor_source_vars) {
      # Build vector of length n_cells: x[matrix_idx] = value
      x <- rep(NA_real_, n_cells)
      x[yr_idx] <- yr_rows[[var_name]]
      
      # Compute neighbor stats using pre-extracted edges
      neighbor_vals <- x[edge_col]
      
      dt_edges <- data.table(row = edge_row, val = neighbor_vals)
      dt_edges <- dt_edges[!is.na(val)]
      
      if (nrow(dt_edges) > 0L) {
        agg <- dt_edges[, .(
          nb_max  = max(val),
          nb_min  = min(val),
          nb_mean = mean(val)
        ), by = row]
        
        # Map aggregated results back: agg$row is matrix index
        # We need to assign to the cell_data rows for this year
        # Create mapping from matrix_idx -> result
        result_max  <- rep(NA_real_, n_cells)
        result_min  <- rep(NA_real_, n_cells)
        result_mean <- rep(NA_real_, n_cells)
        result_max[agg$row]  <- agg$nb_max
        result_min[agg$row]  <- agg$nb_min
        result_mean[agg$row] <- agg$nb_mean
        
        col_max  <- paste0("neighbor_max_", var_name)
        col_min  <- paste0("neighbor_min_", var_name)
        col_mean <- paste0("neighbor_mean_", var_name)
        
        # Assign back to cell_data for this year's rows
        # Use direct row indices for assignment
        row_indices <- which(cell_data$year == yr)
        mat_indices <- cell_data$.mat_idx[row_indices]
        
        set(cell_data, i = row_indices, j = col_max,  value = result_max[mat_indices])
        set(cell_data, i = row_indices, j = col_min,  value = result_min[mat_indices])
        set(cell_data, i = row_indices, j = col_mean, value = result_mean[mat_indices])
      }
    }
    
    elapsed_yr <- (proc.time() - t_yr)[3]
    cat(sprintf("  Year %d done (%.1f sec)\n", yr, elapsed_yr))
  }
  
  elapsed_total <- (proc.time() - t_total)[3]
  cat(sprintf("Total neighbor aggregation: %.1f sec (%.1f min)\n",
              elapsed_total, elapsed_total / 60))
  
  # Clean up helper column
  cell_data[, .mat_idx := NULL]
  
  cell_data
}

# ---- Step 4: Predict with existing Random Forest (UNCHANGED) ----

# Usage:
# cell_data <- run_neighbor_aggregation(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)

# =============================================================================
# ALTERNATIVE: Even faster version using matrix operations for mean,
# and vectorized approach for max/min via sorted-edge trick
# =============================================================================

# If max/min are not needed and only mean is required, the fastest approach is:
#   degree <- rowSums(A)  # or A %*% rep(1, n)
#   neighbor_sum <- A %*% x
#   neighbor_mean <- neighbor_sum / degree
# This runs in milliseconds per variable-year.

# For the full pipeline including max/min, the data.table grouped aggregation
# above is the practical optimum in R without dropping to C++.
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste keys, named vector lookup | Sparse matrix built once from `nb` object |
| **Stats computation** | 6.46M R-level `lapply` iterations | `data.table` grouped aggregation on ~1.37M edge rows × 28 years |
| **Memory** | 6.46M-element list of integer vectors + string keys | One sparse matrix (~22 MB for 1.37M entries) + one 344K vector per var-year |
| **Estimated time** | 86+ hours | **5–15 minutes** |
| **Numerical equivalence** | Baseline | ✅ Identical `max`, `min`, `mean` over non-NA neighbor values |
| **RF model** | — | ✅ Untouched, used only for `predict()` |

**Why this is numerically equivalent:** The sparse matrix encodes exactly the same directed rook-neighbor relationships. For each cell *i* and year *t*, we gather `{x[j] : A[i,j]=1}`, drop NAs, and compute `max`, `min`, `mean` — identical to the original `compute_neighbor_stats`. The `data.table` grouped aggregation is an algebraically equivalent rewriting of the per-row loop.