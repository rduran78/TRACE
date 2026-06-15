 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It creates a list of 6.46 million elements, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The only thing that changes is the variable values attached to those cells each year.

2. **String-key hashing is expensive at scale.** The function creates `paste(id, year)` keys for all 6.46M rows, then for each row, pastes neighbor IDs with that row's year, and looks them up. This is O(n × k) string operations where n = 6.46M and k = average neighbor count (~4 for rook).

3. **`compute_neighbor_stats` iterates over 6.46M list elements with `lapply`.** Each call extracts neighbor values, removes NAs, and computes max/min/mean — all in an interpreted R loop.

4. **The entire process repeats for each of 5 variables**, but the neighbor lookup (the expensive part) is the same every time.

### The Key Insight

- **Static:** The neighbor graph (which cells are neighbors of which) is fixed across all 28 years.
- **Dynamic:** The variable values (ntl, ec, pop_density, def, usd_est_n2) change by year.

Therefore, we should:
- Build the neighbor lookup **once at the cell level** (344K cells, not 6.46M cell-years).
- For each variable, split by year, compute neighbor stats using the **cell-level** neighbor structure via vectorized/matrix operations, then reassemble.

---

## Optimization Strategy

### 1. Cell-Level Neighbor Index (build once, ~344K entries)

Build a simple list mapping each cell's positional index (1..344208) to its neighbors' positional indices. This is a one-time O(344K) operation using the existing `rook_neighbors_unique` nb object directly — no string hashing needed.

### 2. Sparse Adjacency Matrix (build once)

Convert the nb object to a sparse adjacency matrix (`dgCMatrix`). This enables **vectorized** neighbor aggregation via sparse matrix–vector multiplication for the mean, and analogous operations for max and min.

### 3. Year-Sliced Vectorized Computation

For each year and each variable:
- Extract the variable as a vector aligned to cell order.
- **Mean:** One sparse matrix–vector multiply + divide by neighbor count.
- **Max/Min:** Use the sparse matrix structure to compute row-wise max/min efficiently in C++ via a small Rcpp function, or use a grouped approach.

### 4. Complexity Reduction

| Aspect | Before | After |
|---|---|---|
| Lookup construction | 6.46M string-key lookups | 1 sparse matrix (344K × 344K) |
| Stats computation | 6.46M × 5 R-level lapply iterations | 28 × 5 sparse mat-vec ops |
| Estimated time | 86+ hours | **~2–5 minutes** |

---

## Working R Code

```r
library(Matrix)
library(spdep)
library(data.table)

# ==============================================================================
# STEP 1: Build a sparse adjacency matrix from the nb object (ONCE, static)
# ==============================================================================

build_sparse_neighbor_matrix <- function(nb_obj) {
  # nb_obj: spdep nb object, length = number of cells (344,208)
  n <- length(nb_obj)
  
  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove zero-neighbor placeholders (spdep uses integer(0) or 0L)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse binary adjacency matrix (rows = focal cell, cols = neighbor cell)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

# ==============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable across all years
#          using the sparse matrix
# ==============================================================================

compute_neighbor_features_sparse <- function(DT, var_name, W, id_order) {
  # DT:       data.table with columns: id, year, <var_name>
  # W:        sparse adjacency matrix (n_cells x n_cells)
  # id_order: vector of cell IDs in the order matching W's row/col indices
  
  n_cells <- length(id_order)
  
  # Map cell id -> matrix row index (positional index in id_order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add positional index to DT
  DT[, cell_pos := id_to_pos[as.character(id)]]
  
  # Precompute neighbor counts per cell (static)
  neighbor_count <- as.numeric(rowSums(W))  # length n_cells
  
  # Column names for output
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Initialize output columns
  DT[, (col_max)  := NA_real_]
  DT[, (col_min)  := NA_real_]
  DT[, (col_mean) := NA_real_]
  
  # Get the sparse structure for row-wise max/min computation
  # W is in dgCMatrix (column-compressed) format; convert to dgRMatrix or
  # use the dgCMatrix directly. For row operations, dgRMatrix is better.
  # However, we can also work with dgCMatrix by transposing.
  # We'll extract the row-pointer structure.
  
  Wt <- t(W)  # Now Wt is dgCMatrix; column j of Wt = neighbors of cell j
  # Wt@p: column pointers, Wt@i: row indices (= neighbor cell indices)
  
  years <- sort(unique(DT$year))
  
  for (yr in years) {
    # Extract rows for this year
    yr_mask <- DT$year == yr
    
    # Build a full-length vector for this variable, indexed by cell_pos
    # (some cells may be missing for a year; they get NA)
    val_vec <- rep(NA_real_, n_cells)
    
    yr_pos  <- DT$cell_pos[yr_mask]
    yr_vals <- DT[[var_name]][yr_mask]
    val_vec[yr_pos] <- yr_vals
    
    # --- MEAN: sparse matrix-vector multiply ---
    # W %*% val_vec gives sum of neighbor values for each cell
    # Handle NAs: replace NA with 0 for sum, track non-NA count
    val_nona <- val_vec
    val_nona[is.na(val_nona)] <- 0
    
    is_valid <- as.numeric(!is.na(val_vec))
    
    neighbor_sum     <- as.numeric(W %*% val_nona)
    neighbor_nvalid  <- as.numeric(W %*% is_valid)
    
    neighbor_mean_vec <- ifelse(neighbor_nvalid > 0,
                                neighbor_sum / neighbor_nvalid,
                                NA_real_)
    
    # --- MAX and MIN: iterate over sparse structure ---
    # Use Wt (dgCMatrix). For each cell j, neighbors are in 
    # Wt@i[(Wt@p[j]+1):Wt@p[j+1]] (0-based indices)
    neighbor_max_vec <- rep(NA_real_, n_cells)
    neighbor_min_vec <- rep(NA_real_, n_cells)
    
    p_ptr <- Wt@p   # length n_cells + 1, 0-based
    i_idx <- Wt@i   # 0-based row indices
    
    # Vectorized approach: for each cell, gather neighbor values
    # We can do this efficiently by working on the full neighbor value vector
    # and using grouping.
    
    # Build a "neighbor values" vector aligned to the sparse entries
    # i_idx contains the neighbor cell indices (0-based)
    all_neighbor_vals <- val_vec[i_idx + 1L]  # +1 for R's 1-based indexing
    
    # Build a grouping vector: which focal cell does each entry belong to?
    # Cell j owns entries from p_ptr[j]+1 to p_ptr[j+1] (1-based: p_ptr[j+1]+1 to p_ptr[j+1+1])
    n_neighbors_per_cell <- diff(p_ptr)  # length n_cells
    focal_cell_group <- rep(seq_len(n_cells), times = n_neighbors_per_cell)
    
    # Now compute grouped max and min using data.table for speed
    if (length(all_neighbor_vals) > 0) {
      tmp_dt <- data.table(
        focal = focal_cell_group,
        nval  = all_neighbor_vals
      )
      
      # Remove NA neighbor values before aggregation
      tmp_dt <- tmp_dt[!is.na(nval)]
      
      if (nrow(tmp_dt) > 0) {
        agg <- tmp_dt[, .(nmax = max(nval), nmin = min(nval)), by = focal]
        neighbor_max_vec[agg$focal] <- agg$nmax
        neighbor_min_vec[agg$focal] <- agg$nmin
      }
    }
    
    # Write results back to DT for this year's rows
    set(DT, which = yr_mask, j = col_max,  value = neighbor_max_vec[yr_pos])
    set(DT, which = yr_mask, j = col_min,  value = neighbor_min_vec[yr_pos])
    set(DT, which = yr_mask, j = col_mean, value = neighbor_mean_vec[yr_pos])
  }
  
  # Clean up helper column
  DT[, cell_pos := NULL]
  
  return(DT)
}

# ==============================================================================
# STEP 3: Full pipeline — drop-in replacement for the original outer loop
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building sparse adjacency matrix (one-time, static topology)...\n")
  W <- build_sparse_neighbor_matrix(rook_neighbors_unique)
  cat(sprintf("  Sparse matrix: %d x %d, %d non-zero entries\n",
              nrow(W), ncol(W), nnzero(W)))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
    t0 <- proc.time()
    cell_data <- compute_neighbor_features_sparse(
      cell_data, var_name, W, id_order
    )
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds.\n", elapsed))
  }
  
  # Convert back to data.frame if downstream code expects it
  # cell_data <- as.data.frame(cell_data)  # uncomment if needed
  
  return(cell_data)
}

# ==============================================================================
# USAGE (replaces the original outer loop)
# ==============================================================================

cell_data <- run_neighbor_feature_pipeline(
  cell_data, id_order, rook_neighbors_unique
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical estimand** | Max, min, and mean are computed over exactly the same neighbor sets and the same variable values. The sparse matrix encodes the identical rook-neighbor graph. NA handling is preserved. |
| **Trained RF model untouched** | No model retraining occurs. The output columns have the same names and semantics. The prediction call is unchanged. |
| **Static vs. dynamic separation** | The sparse matrix `W` is built **once** from the nb object (static topology). Variable values are sliced **per year** (dynamic data). This is the core redesign. |

## Performance Summary

| Operation | Count | Cost |
|---|---|---|
| Sparse matrix construction | 1 | ~1 second |
| Sparse mat-vec multiply (mean) | 28 years × 5 vars = 140 | ~0.1s each |
| Grouped max/min via data.table | 140 | ~0.3s each |
| **Total estimated wall time** | | **~2–5 minutes** |

This represents a speedup of roughly **1,000–2,500×** over the original 86+ hour implementation, achieved entirely by separating the static neighbor topology from the dynamic yearly variable values.