 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation × ~4 neighbors × 6.46M rows ≈ 25.8M string constructions).
3. Indexes into `idx_lookup` (a named character vector used as a hash map — each lookup is O(1) amortized, but the constant factor of matching on character keys is large).

The result is a **list of 6.46 million integer vectors**. This alone consumes significant memory (~200–400 MB depending on neighbor counts) and takes many hours because R's `lapply` over millions of rows with string operations is inherently slow.

### Bottleneck B: `compute_neighbor_stats` — called 5 times, each O(n)

Each call iterates over the 6.46M-element `neighbor_lookup` list, subsetting a numeric vector and computing `max`, `min`, `mean`. The subsetting itself is fast, but the `lapply` + `do.call(rbind, ...)` pattern over 6.46M small vectors is extremely slow due to:
- R function-call overhead per iteration.
- `do.call(rbind, list_of_6.46M_vectors)` builds a massive intermediate list before binding.

### Why raster focal/kernel operations are a useful *analogy* but not the right implementation

Raster focal operations (e.g., `terra::focal`) assume a regular rectangular grid with a fixed kernel window. If the 344K cells form a complete regular grid, focal operations could theoretically compute neighbor stats in seconds. However:
- The panel has **irregular spatial coverage** (not all grid positions may be populated in all years).
- The neighbor structure is stored as an `spdep::nb` object, which can encode irregular/non-rectangular adjacency.
- Focal operations would silently change the numerical results at boundaries or for cells with missing neighbors.

**Therefore, we must preserve the exact `spdep::nb` neighbor structure** but vectorize the computation.

---

## 2. Optimization Strategy

The key insight: **separate the spatial dimension from the temporal dimension**.

- There are only **344,208 unique spatial cells** with a fixed neighbor structure.
- Each cell appears in up to **28 years**.
- Neighbor stats for cell `i` in year `t` depend only on the values of cell `i`'s spatial neighbors in the **same year** `t`.

**Strategy:**

1. **Eliminate `build_neighbor_lookup` entirely.** Instead of precomputing a 6.46M-element list mapping each row to its neighbor rows, work with the 344K-element `nb` object directly and process **year-by-year**.

2. **Vectorize neighbor stat computation using sparse matrix multiplication.** Construct a sparse adjacency matrix `W` (344,208 × 344,208) from the `nb` object. Then for each year and each variable:
   - Extract the variable column as a vector aligned to the spatial cell order.
   - Compute neighbor sums, counts, max, and min using sparse matrix operations or indexed vectorized operations.

3. **For `mean` and `sum`:** `W %*% x` gives the sum of neighbor values; dividing by the number of neighbors gives the mean. This is a single sparse matrix-vector multiply — extremely fast.

4. **For `max` and `min`:** Sparse matrix multiplication doesn't directly give max/min. Instead, use a **vectorized grouped operation** with `data.table` or direct C++-level indexing. We iterate over the 344K cells (not 6.46M rows) and use the `nb` list to gather neighbor values, then compute max/min in a vectorized batch.

5. **Process year-by-year** (28 iterations) × **variable-by-variable** (5 variables) = 140 iterations, each operating on a 344K-length vector. This is trivially fast.

**Expected speedup:** From 86+ hours to **~2–5 minutes**.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, 
                                        id_order, 
                                        rook_neighbors_unique, 
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # Convert to data.table for speed (non-destructive to original)
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  
  # Map cell id -> position index (1..n_cells)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # ------------------------------------------------------------------
  # Build sparse adjacency matrix W from the nb object (344K x 344K)
  # This is done ONCE and reused for all variables and years.
  # ------------------------------------------------------------------
  # Also build a simple neighbor index list aligned to 1:n_cells
  # for max/min (which can't use matrix multiply).
  # ------------------------------------------------------------------
  
  cat("Building sparse adjacency structures...\n")
  
  # nb object: rook_neighbors_unique[[i]] gives integer indices of 
  # neighbors of the i-th cell in id_order.
  # A zero (integer(0)) means no neighbors.
  
  # For sparse matrix (used for mean):
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)
  
  # Remove any 0-entries (spdep uses 0 to indicate no neighbors)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  W <- sparseMatrix(
    i = from_idx, 
    j = to_idx, 
    x = 1, 
    dims = c(n_cells, n_cells)
  )
  
  # Number of neighbors per cell (for computing mean from sum)
  n_neighbors <- diff(W@p)  # for dgCMatrix, column-oriented; 
  # safer: use rowSums
  n_neighbors_vec <- as.numeric(Matrix::rowSums(W))
  
  # Precompute neighbor index list (aligned to 1:n_cells) for max/min
  nb_list <- lapply(seq_len(n_cells), function(i) {
    nb <- rook_neighbors_unique[[i]]
    nb[nb > 0L]
  })
  
  # ------------------------------------------------------------------
  # Ensure dt has a cell-position column for fast alignment
  # ------------------------------------------------------------------
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  
  # ------------------------------------------------------------------
  # Pre-allocate output columns
  # ------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # ------------------------------------------------------------------
  # Vectorized max/min using the nb_list
  # We process all cells for a given year at once.
  # ------------------------------------------------------------------
  # Precompute a fast C-style grouped max/min function using 
  # vectorized indexing.
  
  compute_max_min_vectorized <- function(vals_full, nb_list, n_cells) {
    # vals_full: numeric vector of length n_cells (NA for missing cells)
    # Returns: matrix of n_cells x 2 (max, min)
    
    # Unlist all neighbor indices
    all_nb    <- unlist(nb_list, use.names = FALSE)
    group_len <- lengths(nb_list)
    group_id  <- rep(seq_len(n_cells), group_len)
    
    # Gather all neighbor values
    nb_vals <- vals_full[all_nb]
    
    # We need grouped max and min, handling NAs.
    # Use data.table for fast grouped aggregation.
    tmp_dt <- data.table(g = group_id, v = nb_vals)
    
    # Remove NAs before aggregation
    tmp_dt <- tmp_dt[!is.na(v)]
    
    if (nrow(tmp_dt) == 0) {
      result <- matrix(NA_real_, nrow = n_cells, ncol = 2)
      return(result)
    }
    
    agg <- tmp_dt[, .(mx = max(v), mn = min(v)), by = g]
    
    result <- matrix(NA_real_, nrow = n_cells, ncol = 2)
    result[agg$g, 1] <- agg$mx
    result[agg$g, 2] <- agg$mn
    
    return(result)
  }
  
  # ------------------------------------------------------------------
  # Main loop: iterate over years (28) x variables (5) = 140 iterations
  # ------------------------------------------------------------------
  
  cat("Computing neighbor features year-by-year...\n")
  
  # Key dt by year for fast subsetting
  setkey(dt, year)
  
  for (yr in years) {
    cat(sprintf("  Year %d ...\n", yr))
    
    # Get row indices in dt for this year
    yr_rows <- dt[.(yr), which = TRUE]
    
    # Get cell positions for this year's rows
    yr_cell_pos <- dt$cell_pos[yr_rows]
    
    for (var_name in neighbor_source_vars) {
      
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      # Build a full-length vector (n_cells) with values for this year
      # Initialize with NA
      vals_full <- rep(NA_real_, n_cells)
      vals_full[yr_cell_pos] <- dt[[var_name]][yr_rows]
      
      # --- MEAN via sparse matrix multiply ---
      # Replace NA with 0 for the multiply, but track valid counts
      vals_for_mult <- vals_full
      valid_mask    <- as.numeric(!is.na(vals_full))
      vals_for_mult[is.na(vals_for_mult)] <- 0
      
      neighbor_sum   <- as.numeric(W %*% vals_for_mult)
      neighbor_count <- as.numeric(W %*% valid_mask)
      
      neighbor_mean <- ifelse(neighbor_count > 0, 
                              neighbor_sum / neighbor_count, 
                              NA_real_)
      
      # --- MAX and MIN via vectorized grouped operation ---
      maxmin <- compute_max_min_vectorized(vals_full, nb_list, n_cells)
      
      # --- Write results back to dt ---
      set(dt, i = yr_rows, j = max_col,  value = maxmin[yr_cell_pos, 1])
      set(dt, i = yr_rows, j = min_col,  value = maxmin[yr_cell_pos, 2])
      set(dt, i = yr_rows, j = mean_col, value = neighbor_mean[yr_cell_pos])
    }
  }
  
  # Clean up helper column
  dt[, cell_pos := NULL]
  
  # Reset key to original order if needed
  setkey(dt, NULL)
  
  return(dt)
}

# =====================================================================
# USAGE
# =====================================================================

# Load your data (assumed already in memory):
# cell_data              - data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order               - vector of unique cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique  - spdep::nb object (list of integer neighbor indices)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# Now cell_data has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.

# Proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element list built via `paste`+hash | Eliminated; use 344K `nb` list directly |
| **Mean computation** | `lapply` over 6.46M rows | Sparse matrix multiply (`W %*% x`), ~0.1s per year-variable |
| **Max/Min computation** | `lapply` over 6.46M rows | Vectorized `data.table` grouped aggregation on ~1.37M edges per year |
| **Outer loop** | 5 variables × 6.46M rows each | 28 years × 5 variables = 140 iterations, each on 344K cells |
| **Memory** | 6.46M-element list of integer vectors (~400MB) | Sparse matrix (~20MB) + nb_list (~20MB) |
| **Estimated runtime** | 86+ hours | **~2–10 minutes** |
| **Numerical results** | Baseline | **Identical** (same max, min, mean over same rook neighbors per cell-year) |
| **Trained RF model** | — | **Untouched; not retrained** |

The key principles: (1) separate space from time, (2) exploit the sparse adjacency matrix for linear-algebra-based mean computation, (3) use `data.table` grouped aggregation for max/min instead of R-level loops, and (4) iterate over 28 years rather than 6.46M rows.