 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It iterates over all ~6.46 million rows, performing string-pasting and hash lookups for every single row. But the neighbor *relationships* are purely spatial — cell A neighbors cell B regardless of year. The only thing that changes across years is which *row* holds cell B's data for a given year. This means the function does **O(rows × avg_neighbors)** string operations (~6.46M × ~4 ≈ 25.8M string-paste + hash lookups), producing a list of 6.46 million integer vectors.

2. **`compute_neighbor_stats` then indexes into the full data vector using these row indices.** This is called 5 times (once per variable), each time iterating over 6.46M list elements via `lapply`. The per-element work is small, but the overhead of 6.46M R-level function calls × 5 variables ≈ 32.3M `lapply` iterations is enormous.

3. **Memory bloat.** The `neighbor_lookup` list stores ~6.46M integer vectors. Since each cell has ~4 rook neighbors on average, this is ~25.8M integers, but the R list overhead per element (~128 bytes for a length-4 integer vector including the list slot) means ~800 MB+ just for the lookup structure.

### The Key Insight

The neighbor graph is **static across years**. Cell *i*'s neighbors are always the same set of cells. Only the variable values change year to year. Therefore:

- **Build the neighbor topology once at the cell level** (344,208 cells, not 6.46M rows).
- **Compute neighbor stats per year** by slicing the data by year, then using the cell-level neighbor index to gather values from a cell-indexed vector (not a row-indexed one).

This reduces the lookup construction from 6.46M iterations to 344,208, and makes the stats computation a simple matrix-column operation per year.

---

## Optimization Strategy

### 1. Build a cell-level neighbor lookup once (static topology)

Convert `rook_neighbors_unique` (an `nb` object, already cell-indexed) into a compact structure: for each cell index `c`, store the integer vector of neighbor cell indices. This is essentially what `rook_neighbors_unique` already is — an `nb` object is a list of integer vectors. We just need a fast mapping from cell ID to sequential index.

### 2. Reshape data for fast year-wise access

Create a mapping from `(cell_index, year)` → row index. Since we have 344,208 cells × 28 years, we can use a matrix of dimension `(n_cells, n_years)` storing row indices. This allows O(1) lookup.

### 3. Vectorized neighbor stats via matrix operations

For each variable:
- Extract the variable values into a matrix of shape `(n_cells, n_years)`.
- For each cell, gather its neighbors' rows from this matrix.
- Compute max, min, mean across neighbors for each cell-year.

We can do this extremely efficiently using **CSR-style sparse matrix multiplication** (for mean) and analogous operations for max/min, or by using a pre-flattened neighbor vector approach with `data.table` or vectorized R.

### 4. Estimated speedup

| Aspect | Before | After |
|---|---|---|
| Lookup construction | 6.46M string ops | 0 (reuse `nb` object directly) |
| Stats iterations | 6.46M × 5 = 32.3M | 344K × 28 × 5 = 48.2M cell-year-var, but vectorized |
| Memory for lookup | ~800 MB | ~22 MB (cell-level `nb`) |
| Expected wall time | 86+ hours | **~2–10 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the static-vs-changing distinction:
#   STATIC:  neighbor topology (which cells neighbor which)
#   CHANGING: variable values (which change by year)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          neighbor_source_vars,
                                          id_order,
                                          rook_neighbors_unique) {
  # -------------------------------------------------------------------------
  # STEP 0: Convert to data.table for speed (non-destructive)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # -------------------------------------------------------------------------
  # STEP 1: BUILD STATIC CELL-LEVEL STRUCTURES (done once, topology only)
  # -------------------------------------------------------------------------
  
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
  # rook_neighbors_unique[[c]] gives the integer indices (into id_order) of
  # cell c's neighbors. This is already a cell-level neighbor lookup.
  
  n_cells <- length(id_order)
  
  # Map cell IDs to their sequential index in id_order (1..n_cells)
  cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Build flat CSR-like representation of the neighbor graph for vectorized ops
  # neighbor_offsets[c] = start position in neighbor_flat for cell c
  # neighbor_flat = concatenated neighbor cell indices
  neighbor_lengths <- vapply(rook_neighbors_unique, length, integer(1))
  total_edges      <- sum(neighbor_lengths)
  neighbor_flat    <- unlist(rook_neighbors_unique, use.names = FALSE)
  # Compute offsets (1-indexed start positions)
  neighbor_offsets <- c(0L, cumsum(neighbor_lengths))
  # neighbor_offsets has length n_cells + 1
  # Cell c's neighbors are neighbor_flat[(neighbor_offsets[c]+1):neighbor_offsets[c+1]]
  
  # -------------------------------------------------------------------------
  # STEP 2: BUILD CELL-INDEX COLUMN IN DATA
  # -------------------------------------------------------------------------
  
  # Add cell sequential index to each row
  dt[, cell_idx := cell_id_to_idx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))
  
  # -------------------------------------------------------------------------
  # STEP 3: FOR EACH VARIABLE, COMPUTE NEIGHBOR MAX, MIN, MEAN
  # -------------------------------------------------------------------------
  
  # Strategy: For each variable, build a matrix (n_cells x n_years) of values.
  # Then use the flat neighbor structure to compute stats vectorized.
  
  # Pre-sort dt by (cell_idx, year) for predictable matrix filling
  setkey(dt, cell_idx, year)
  dt[, year_idx := year_to_col[as.character(year)]]
  
  for (var_name in neighbor_source_vars) {
    
    message(sprintf("Computing neighbor stats for: %s", var_name))
    
    # Build value matrix: n_cells x n_years
    # Initialize with NA
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Fill from data
    val_mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # For each year, compute neighbor max/min/mean using vectorized operations
    # Result matrices
    nmax_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nmin_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nmean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Process year by year (28 iterations — trivial loop overhead)
    for (y_idx in seq_len(n_years)) {
      
      # Current year's values for all cells: length n_cells
      year_vals <- val_mat[, y_idx]
      
      # Gather all neighbor values using the flat neighbor index
      # neighbor_flat contains cell indices; look up their values for this year
      neighbor_vals_all <- year_vals[neighbor_flat]
      # neighbor_vals_all has length = total_edges
      
      # Now compute per-cell stats using the CSR structure
      # We use a fast C-level grouped operation via tapply or, better,
      # a vectorized approach with rep + group-by
      
      # Create cell-id vector corresponding to each entry in neighbor_flat
      cell_rep <- rep.int(seq_len(n_cells), times = neighbor_lengths)
      
      # Remove NA neighbor values
      valid <- !is.na(neighbor_vals_all)
      
      if (any(valid)) {
        valid_cells <- cell_rep[valid]
        valid_vals  <- neighbor_vals_all[valid]
        
        # Use data.table for fast grouped aggregation
        agg_dt <- data.table(cell = valid_cells, val = valid_vals)
        agg <- agg_dt[, .(nmax = max(val), nmin = min(val), nmean = mean(val)),
                       by = cell]
        
        nmax_mat[agg$cell, y_idx]  <- agg$nmax
        nmin_mat[agg$cell, y_idx]  <- agg$nmin
        nmean_mat[agg$cell, y_idx] <- agg$nmean
      }
      
      # Cells with neighbor_lengths == 0 or all-NA neighbors remain NA (correct)
    }
    
    # Write results back to dt using the (cell_idx, year_idx) mapping
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := nmax_mat[cbind(cell_idx, year_idx)]]
    dt[, (min_col)  := nmin_mat[cbind(cell_idx, year_idx)]]
    dt[, (mean_col) := nmean_mat[cbind(cell_idx, year_idx)]]
    
    message(sprintf("  Done: %s", var_name))
  }
  
  # -------------------------------------------------------------------------
  # STEP 4: CLEAN UP AND RETURN
  # -------------------------------------------------------------------------
  
  # Remove helper columns
  dt[, c("cell_idx", "year_idx") := NULL]
  
  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  neighbor_source_vars = neighbor_source_vars,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique
)

# The pre-trained Random Forest model is used AFTER this step, unchanged.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical estimand** | For each `(cell, year, variable)` triple, we gather the exact same set of neighbor cell values for that year and compute the identical `max`, `min`, `mean`. No approximation. |
| **Pre-trained RF unchanged** | The code only computes input features. The model object is never touched. Column names (`neighbor_max_ntl`, etc.) match the originals. |
| **Column naming convention** | Uses `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` — adjust the prefix/suffix pattern if your original `compute_and_add_neighbor_features` used a different convention. |

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-paste + hash lookups | **Eliminated** (reuse `nb` object) |
| Core loop iterations | 6.46M × 5 R-level `lapply` calls | 28 × 5 = **140** vectorized year-variable passes |
| Peak memory for lookup | ~800 MB (list of 6.46M vectors) | ~22 MB (flat integer vector + offsets) |
| Estimated wall time | **86+ hours** | **~2–10 minutes** |

The fundamental insight — **separate static topology from dynamic attributes** — eliminates millions of redundant string operations and replaces millions of R-level function calls with a small number of vectorized grouped aggregations.