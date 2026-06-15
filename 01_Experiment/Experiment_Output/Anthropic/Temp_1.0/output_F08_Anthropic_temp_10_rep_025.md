 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` conflates two fundamentally different structures:

1. **Static topology**: Which cells are neighbors of which other cells — this never changes across years.
2. **Dynamic data**: The variable values attached to each cell, which change every year.

The current implementation builds a **row-level** neighbor lookup across the entire 6.46-million-row panel dataset. For each of the ~6.46M rows, it:
- Looks up the cell's neighbors by ID.
- Pastes neighbor IDs with the current row's year to form string keys.
- Does a named-vector lookup (`idx_lookup[neighbor_keys]`) against a 6.46M-element named character vector.

This means:
- `idx_lookup` is a **6.46M-element named character vector** — named lookups on this are O(n) hash probes per call, repeated ~6.46M times.
- The `lapply` over 6.46M rows with string pasting and named-vector indexing is catastrophically slow.
- The entire process is repeated **5 times** (once per neighbor source variable), even though the topology is the same.
- `compute_neighbor_stats` then does another `lapply` over 6.46M rows.

**Total work**: ~6.46M × 5 iterations of string operations, named lookups, and per-row R function calls. This explains the 86+ hour estimate.

## Optimization Strategy

**Key insight**: Separate the static neighbor graph from the year-varying data. Compute neighbor statistics **per year** using the cell-level (not row-level) neighbor graph, operating on vectors/matrices rather than row-by-row string lookups.

**Steps:**

1. **Build a cell-level neighbor lookup once** — a simple list mapping each cell's positional index (1..344,208) to the positional indices of its neighbors. This is just `rook_neighbors_unique` itself (an `nb` object already has this structure). Cost: essentially free.

2. **For each year**, extract the column vector for each variable, subset to that year's cells (all 344,208 cells in order), and compute neighbor max/min/mean using vectorized operations over the cell-level neighbor list.

3. **Use a pre-sorted data.table** to guarantee that within each year, cells appear in the same positional order as `id_order`, so that the static neighbor indices can directly index into the year's value vector.

4. **Compute all 5 variables' neighbor stats in one pass per year** to maximize cache locality.

This reduces the problem from 6.46M string-lookup iterations to 28 years × 344,208 cells × simple integer-indexed vector operations — roughly a **500–1000× speedup**. Expected runtime: **1–3 minutes** on a standard laptop.

The numerical estimand is preserved exactly: for each cell-year, the neighbor max, min, and mean are computed over the same rook neighbors' values for that same year.

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation.
#' Separates static topology from dynamic (year-varying) data.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs defining the canonical positional order (length = n_cells)
#' @param rook_nb          spdep nb object (list of length n_cells); rook_nb[[i]] gives integer
#'                         positional indices of neighbors of cell i (in id_order space)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor_{var}_max, _min, _mean columns appended
compute_neighbor_features_optimized <- function(cell_data,
                                                 id_order,
                                                 rook_nb,
                                                 neighbor_source_vars) {
  
  # Convert to data.table if needed (by reference if already one)
  dt <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  # --- Step 1: Build cell-position mapping (static, done once) ---
  # Map cell ID -> positional index in id_order (1..n_cells)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Step 2: Ensure data is keyed by (year, id) and sorted for fast subsetting ---
  # Add positional index to each row
  dt[, cell_pos := id_to_pos[as.character(id)]]
  setkey(dt, year, cell_pos)
  
  # --- Step 3: Pre-compute neighbor lengths and a flattened neighbor structure for vectorized ops ---
  # For each cell position i, rook_nb[[i]] gives neighbor positions.
  # We flatten this into two vectors for fast grouped operations.
  nb_lengths <- vapply(rook_nb, length, integer(1))
  
  # "owner" index: which cell each neighbor entry belongs to
  # "neighbor" index: the neighbor's positional index
  owner_idx    <- rep(seq_len(n_cells), nb_lengths)
  neighbor_idx <- unlist(rook_nb)
  
  # Cells with no neighbors (if any)
  has_neighbors <- nb_lengths > 0
  
  # --- Step 4: Pre-allocate output columns ---
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_", var_name, "_max")
    min_col  <- paste0("neighbor_", var_name, "_min")
    mean_col <- paste0("neighbor_", var_name, "_mean")
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # --- Step 5: Loop over years (only 28 iterations) ---
  for (yr in years) {
    
    # Extract this year's slice — already sorted by cell_pos due to setkey
    yr_mask <- dt$year == yr
    dt_yr   <- dt[yr_mask]
    
    # Sanity: dt_yr should have n_cells rows, sorted by cell_pos
    # (If some cells are missing for some years, we handle that below)
    
    # Build a value vector of length n_cells for this year.
    # If data is complete (all cells present every year), dt_yr$cell_pos == 1:n_cells
    # and we can directly use the column. Otherwise, map into a full-length vector.
    
    complete_year <- (nrow(dt_yr) == n_cells) && all(dt_yr$cell_pos == seq_len(n_cells))
    
    for (var_name in neighbor_source_vars) {
      
      max_col  <- paste0("neighbor_", var_name, "_max")
      min_col  <- paste0("neighbor_", var_name, "_min")
      mean_col <- paste0("neighbor_", var_name, "_mean")
      
      # Build the full cell-indexed value vector for this year
      if (complete_year) {
        vals <- dt_yr[[var_name]]
      } else {
        vals <- rep(NA_real_, n_cells)
        vals[dt_yr$cell_pos] <- dt_yr[[var_name]]
      }
      
      # --- Vectorized neighbor stat computation ---
      # Gather all neighbor values at once
      neighbor_vals <- vals[neighbor_idx]  # length = total number of directed neighbor edges
      
      # We need max, min, sum, count per owner cell, handling NAs.
      # Use data.table's fast grouped aggregation on the flattened structure.
      agg_dt <- data.table(
        owner = owner_idx,
        nval  = neighbor_vals
      )
      
      # Remove NA neighbor values before aggregation
      agg_dt <- agg_dt[!is.na(nval)]
      
      if (nrow(agg_dt) > 0) {
        stats <- agg_dt[, .(
          nb_max  = max(nval),
          nb_min  = min(nval),
          nb_mean = mean(nval)
        ), by = owner]
        
        # Map results back into full cell-length vectors
        result_max  <- rep(NA_real_, n_cells)
        result_min  <- rep(NA_real_, n_cells)
        result_mean <- rep(NA_real_, n_cells)
        
        result_max[stats$owner]  <- stats$nb_max
        result_min[stats$owner]  <- stats$nb_min
        result_mean[stats$owner] <- stats$nb_mean
      } else {
        result_max  <- rep(NA_real_, n_cells)
        result_min  <- rep(NA_real_, n_cells)
        result_mean <- rep(NA_real_, n_cells)
      }
      
      # Write results back to the main data.table for this year's rows
      if (complete_year) {
        set(dt, which = which(yr_mask), j = max_col,  value = result_max)
        set(dt, which = which(yr_mask), j = min_col,  value = result_min)
        set(dt, which = which(yr_mask), j = mean_col, value = result_mean)
      } else {
        # Map cell_pos back to rows
        yr_rows  <- which(yr_mask)
        yr_cpos  <- dt$cell_pos[yr_rows]
        set(dt, which = yr_rows, j = max_col,  value = result_max[yr_cpos])
        set(dt, which = yr_rows, j = min_col,  value = result_min[yr_cpos])
        set(dt, which = yr_rows, j = mean_col, value = result_mean[yr_cpos])
      }
    }
    
    # Optional: print progress
    message(sprintf("Year %d complete.", yr))
  }
  
  # Clean up helper column
  dt[, cell_pos := NULL]
  
  return(dt)
}
```

### Drop-in replacement for the outer loop

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (~1-3 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_nb              = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Proceed with prediction using the pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | Rook neighbors from `rook_neighbors_unique` | Same `nb` object, same positional indices |
| Year scoping | String key `paste(id, year)` matches neighbors within same year | Explicit year-loop ensures same-year matching |
| Aggregation functions | `max()`, `min()`, `mean()` with NA removal | Identical functions with `!is.na()` filtering |
| Output columns | `neighbor_{var}_max/min/mean` | Identical column names and semantics |
| RF model | Untouched | Untouched — only the feature engineering step is changed |

## Performance Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string pastes + named-vector lookups | Zero — uses `nb` object directly |
| Per-variable computation | 6.46M R-level `lapply` iterations | 28 iterations × vectorized `data.table` grouped aggregation |
| Total R function calls | ~32.3M (6.46M × 5 vars) | 140 (28 years × 5 vars) |
| Memory for lookup | 6.46M-element list of integer vectors | Two integer vectors (owner + neighbor), ~11M elements total |
| Estimated wall time | 86+ hours | **1–3 minutes** |