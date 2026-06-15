 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`paste` + `idx_lookup[neighbor_keys]`). This is O(N×K) with enormous constant factors from R's string hashing on millions of keys.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in an `lapply`**, calling `max`, `min`, `mean` individually per node. R-level loops over millions of elements are inherently slow.

3. **The neighbor lookup is year-coupled but the graph topology is year-invariant.** The rook adjacency is purely spatial (344,208 cells), yet the lookup is rebuilt across all 6.46M cell-year rows, redundantly replicating the same spatial structure 28 times.

**Key insight:** The adjacency graph is static across years. The topology needs to be built once over 344K cells, and then neighbor aggregation can be performed **per-year** using sparse matrix–vector multiplication, which is O(nnz) and runs in optimized C/Fortran code via the `Matrix` package.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 × 344,208, ~1.37M nonzeros). This is tiny in memory (~16 MB).

2. **For each year, extract the variable vector for all cells, then compute neighbor stats via sparse matrix operations:**
   - **Mean:** `A %*% x / A %*% 1` (sum of neighbor values / count of neighbors, handling NAs).
   - **Max / Min:** Use a single pass over the CSC/CSR structure of the sparse matrix — far faster than millions of R-level list lookups.

3. **Vectorize NA handling** using sparse matrix arithmetic: replace NAs with 0 for summation, track valid counts with a separate sparse multiply.

4. **Process year-by-year** to keep memory bounded (~344K vectors, trivially small).

**Expected speedup:** From 86+ hours to **~5–15 minutes** on the same laptop. The sparse matrix multiply is O(nnz) per variable-year (~1.37M operations), and we have 5 variables × 28 years = 140 such operations for each of max/min/mean.

## Working R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 1: Build sparse adjacency matrix from spdep nb object (done ONCE)
# =============================================================================
build_sparse_adjacency <- function(nb_obj) {
  # nb_obj is a list of length N; nb_obj[[i]] gives integer indices of neighbors of cell i
  # (0L means no neighbors in spdep convention)
  n <- length(nb_obj)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # spdep uses 0L for "no neighbors" — remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Row i, Col j = 1 means j is a neighbor of i (i.e., we aggregate j's value for node i)
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

# =============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable across all years
#          using sparse matrix operations — numerically equivalent to original
# =============================================================================
compute_neighbor_features_sparse <- function(dt, var_name, A, id_to_row) {
  # Pre-allocate output columns
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  n_cells <- nrow(A)
  years   <- sort(unique(dt$year))
  
  # CSC structure for fast column-wise access (for max/min)
  # A is n x n: A[i,j]=1 means j is neighbor of i
  # For row-wise iteration (neighbors of i), we use the transpose in CSC = original in CSR
  At <- t(A)  # Now At in CSC format: column i of At = row i of A = neighbors of i
  p  <- At@p  # column pointers (0-indexed)
  j_idx <- At@i  # row indices (0-indexed) = neighbor cell indices for each node
  
  for (yr in years) {
    # Get row indices in dt for this year
    yr_rows <- which(dt$year == yr)
    
    # Map cell ids to spatial row index
    cell_ids <- dt$id[yr_rows]
    spatial_idx <- id_to_row[as.character(cell_ids)]
    
    # Build full-length vector aligned to spatial grid (NA for missing cells)
    x_full <- rep(NA_real_, n_cells)
    x_full[spatial_idx] <- dt[[var_name]][yr_rows]
    
    # --- MEAN via sparse matrix multiply ---
    # Handle NAs: replace with 0 for sum, track counts
    x_nona <- x_full
    x_nona[is.na(x_nona)] <- 0
    valid_mask <- as.double(!is.na(x_full))
    
    neighbor_sum   <- as.numeric(A %*% x_nona)
    neighbor_count <- as.numeric(A %*% valid_mask)
    
    n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN via CSC traversal (vectorized in C-level sparse structure) ---
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
    
    # Process each cell using the sparse structure
    # At column i contains the neighbor indices of cell i
    # We do this in vectorized chunks
    
    # Expand all neighbor pairs
    # For each cell i (0-indexed in p), neighbors are j_idx[(p[i]+1):p[i+1]]
    # We vectorize this:
    
    # Number of neighbors per cell
    n_neighbors <- diff(p)  # length n_cells
    
    # Cell index for each entry in j_idx
    cell_rep <- rep(seq_len(n_cells), times = n_neighbors)
    neighbor_spatial <- j_idx + 1L  # convert to 1-indexed
    
    # Get neighbor values
    neighbor_vals <- x_full[neighbor_spatial]
    
    # Use data.table for fast grouped max/min/mean
    if (length(cell_rep) > 0) {
      agg_dt <- data.table(cell = cell_rep, val = neighbor_vals)
      agg_dt <- agg_dt[!is.na(val)]
      
      if (nrow(agg_dt) > 0) {
        stats <- agg_dt[, .(vmax = max(val), vmin = min(val)), by = cell]
        n_max[stats$cell] <- stats$vmax
        n_min[stats$cell] <- stats$vmin
      }
    }
    
    # Write back to dt for this year's rows
    set(dt, i = yr_rows, j = col_max,  value = n_max[spatial_idx])
    set(dt, i = yr_rows, j = col_min,  value = n_min[spatial_idx])
    set(dt, i = yr_rows, j = col_mean, value = n_mean[spatial_idx])
  }
  
  dt
}

# =============================================================================
# STEP 3: Main pipeline
# =============================================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  
  # Convert to data.table for performance (non-destructive if already data.table)
  dt <- as.data.table(cell_data)
  
  # --- Build spatial index mapping: cell id -> row in adjacency matrix ---
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Build sparse adjacency matrix ONCE ---
  message("Building sparse adjacency matrix...")
  A <- build_sparse_adjacency(rook_neighbors_unique)
  message(sprintf("  Adjacency matrix: %d x %d, %d nonzeros", 
                  nrow(A), ncol(A), nnzero(A)))
  
  # --- Compute neighbor features for each source variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    dt <- compute_neighbor_features_sparse(dt, var_name, A, id_to_row)
  }
  
  # --- Predict using the pre-trained Random Forest (no retraining) ---
  message("Generating predictions with pre-trained Random Forest...")
  predictions <- predict(rf_model, newdata = dt)
  dt[, prediction := predictions]
  
  message("Done.")
  dt
}

# =============================================================================
# Usage (assuming objects are already loaded):
# =============================================================================
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized |
|-----------|----------|-----------|
| **Max** | `max(vals[idx])` after removing NAs | `data.table` grouped `max` over the same neighbor values, same NA exclusion |
| **Min** | `min(vals[idx])` after removing NAs | `data.table` grouped `min` over the same neighbor values, same NA exclusion |
| **Mean** | `mean(vals[idx])` after removing NAs | `sum(valid_neighbor_vals) / count(valid_neighbor_vals)` via sparse multiply — algebraically identical |
| **No neighbors / all-NA neighbors** | Returns `c(NA, NA, NA)` | Returns `NA` for all three — identical |

The sparse matrix `A` encodes exactly the same directed rook-neighbor relationships as `rook_neighbors_unique`. The per-year slicing ensures each cell-year sees only same-year neighbor values, exactly as the original `paste(id, year)` key matching does.

## Performance Estimate

| Component | Original | Optimized |
|-----------|----------|-----------|
| Neighbor lookup construction | ~hours (6.46M string keys) | ~2 seconds (sparse matrix from integer lists) |
| Per-variable aggregation | ~17 hours (6.46M `lapply` iterations) | ~30 seconds (28 sparse multiplies + grouped `data.table` ops) |
| Total (5 variables) | **86+ hours** | **~5–10 minutes** |
| Peak RAM | Potentially >16 GB (list of 6.46M vectors + string keys) | ~2–3 GB (sparse matrix is ~16 MB; data.table in-place) |