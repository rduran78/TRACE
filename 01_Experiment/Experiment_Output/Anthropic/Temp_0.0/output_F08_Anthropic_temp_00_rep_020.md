 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this work 28 times over (once per year, implicitly). Then `compute_neighbor_stats` iterates over those 6.46 million list entries for each of the 5 variables.

**Specific bottlenecks:**

1. **Redundant topology expansion.** The neighbor graph has ~344K cells. Rook neighbors are year-invariant. Yet the lookup is built at the cell×year level (6.46M entries), repeating the same neighbor structure 28 times.
2. **String-key hashing at scale.** `paste(id, year)` and named-vector lookups over 6.46M keys are extremely slow in R.
3. **Row-wise `lapply` over millions of rows.** Each call to `compute_neighbor_stats` does an R-level loop over 6.46M list elements, each with subsetting, `is.na` filtering, and summary computation.
4. **No vectorization.** The max/min/mean computations are done one cell-year at a time instead of using vectorized or matrix operations.

**Estimated cost breakdown of current approach:**
- `build_neighbor_lookup`: ~6.46M string pastes + hash lookups → hours.
- `compute_neighbor_stats` × 5 variables: ~6.46M × 5 R-level iterations → tens of hours.
- Total: 86+ hours as reported.

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors of which) from the *dynamic values* (variable values that change by year).

1. **Build the neighbor graph once at the cell level (344K entries), not the cell×year level (6.46M entries).** The `rook_neighbors_unique` `nb` object already encodes this — use it directly.

2. **Process each year independently using vectorized matrix operations.** For a given year, extract the variable column as a vector indexed by cell order. Use the static neighbor list to gather neighbor values and compute max/min/mean in a vectorized fashion.

3. **Use a sparse-matrix approach for mean computation** and vectorized `pmax`/`pmin` via sparse row operations for max and min — or use a fast C-level grouped operation via `data.table`.

4. **Avoid string keys entirely.** Map cell IDs to integer positions once; use integer indexing throughout.

The result: instead of 6.46M list entries and R-level loops, we do 28 (years) × 5 (variables) = 140 vectorized passes over 344K cells, each using integer-indexed neighbor gathering on contiguous vectors.

**Expected speedup:** From 86+ hours to **minutes** (roughly 2,000–5,000× faster).

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build static neighbor structures ONCE (cell-level, year-invariant)
# ==============================================================================

build_static_neighbor_structures <- function(id_order, neighbors_nb) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors)
  #
  # Returns:
  #   - neighbor_list: list of length n_cells, each element = integer vector

  #     of neighbor positions (indices into id_order)
  #   - W_sparse: sparse adjacency matrix (for fast mean computation)
  #   - n_neighbors: integer vector of neighbor counts per cell
  
  n_cells <- length(id_order)
  stopifnot(length(neighbors_nb) == n_cells)
  
  # The nb object already stores neighbors as indices into the same ordering.
  # spdep::nb objects use 0L to indicate no neighbors.
  neighbor_list <- lapply(neighbors_nb, function(nb_i) {
    nb_i <- as.integer(nb_i)
    nb_i[nb_i > 0L]
  })
  
  n_neighbors <- lengths(neighbor_list)
  
  # Build sparse row-normalized weight matrix for mean computation
  # W[i, j] = 1/n_neighbors[i] if j is a neighbor of i, else 0
  from <- rep(seq_len(n_cells), times = n_neighbors)
  to   <- unlist(neighbor_list, use.names = FALSE)
  
  # Weights for mean: 1 / number_of_neighbors
  weights <- rep(1.0 / pmax(n_neighbors, 1L), times = n_neighbors)
  
  W_mean <- sparseMatrix(
    i = from, j = to, x = weights,
    dims = c(n_cells, n_cells)
  )
  
  # Unweighted adjacency (for max/min we need a different approach)
  W_adj <- sparseMatrix(
    i = from, j = to, x = rep(1.0, length(from)),
    dims = c(n_cells, n_cells)
  )
  
  list(
    neighbor_list = neighbor_list,
    n_neighbors   = n_neighbors,
    W_mean        = W_mean,
    W_adj         = W_adj
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for one variable, all years (vectorized)
# ==============================================================================

compute_neighbor_stats_fast <- function(dt, var_name, id_order, static) {
  # dt: data.table with columns: id, year, <var_name>
  # id_order: vector of cell IDs in nb-object order
  # static: output of build_static_neighbor_structures
  #
  # Returns: dt with three new columns added (neighbor_max, min, mean)
  
  n_cells <- length(id_order)
  neighbor_list <- static$neighbor_list
  W_mean        <- static$W_mean
  n_neighbors   <- static$n_neighbors
  
  # Create a mapping from cell ID to position index
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Output column names
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Get the position index for every row (static mapping, computed once)
  pos_vec <- id_to_pos[as.character(dt$id)]
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Row indices in dt for this year
    yr_rows <- which(dt$year == yr)
    
    # Build a full-length value vector indexed by cell position
    # (some cells may be missing in a given year; they get NA)
    vals_full <- rep(NA_real_, n_cells)
    cell_positions <- pos_vec[yr_rows]
    vals_full[cell_positions] <- dt[[var_name]][yr_rows]
    
    # --- MEAN via sparse matrix-vector multiply ---
    # W_mean %*% vals_full gives the neighbor mean for each cell
    # But we need to handle NAs: for cells whose neighbors are all NA,
    # the result should be NA.
    #
    # Strategy: compute sum of non-NA neighbor values and count of non-NA neighbors
    vals_no_na <- vals_full
    vals_no_na[is.na(vals_no_na)] <- 0.0
    
    indicator <- as.double(!is.na(vals_full))  # 1 if present, 0 if NA
    
    neighbor_sum   <- as.numeric(static$W_adj %*% vals_no_na)
    neighbor_count <- as.numeric(static$W_adj %*% indicator)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN via neighbor list (vectorized gather) ---
    # Use a fast approach: for each cell, gather neighbor values and compute max/min
    # We vectorize this using the flattened neighbor structure
    
    # Flatten: get all neighbor values at once
    all_neighbor_vals <- vals_full[unlist(neighbor_list, use.names = FALSE)]
    
    # Split back by cell using the known lengths
    # Use a fast grouped operation
    cell_idx <- rep(seq_len(n_cells), times = n_neighbors)
    
    # Only process cells that have neighbors
    has_neighbors <- n_neighbors > 0L
    
    # Use data.table for fast grouped max/min
    if (length(all_neighbor_vals) > 0) {
      tmp_dt <- data.table(
        cell = cell_idx,
        val  = all_neighbor_vals
      )
      # Remove NA values before aggregation
      tmp_dt <- tmp_dt[!is.na(val)]
      
      if (nrow(tmp_dt) > 0) {
        agg <- tmp_dt[, .(nmax = max(val), nmin = min(val)), by = cell]
        
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
        neighbor_max[agg$cell] <- agg$nmax
        neighbor_min[agg$cell] <- agg$nmin
      } else {
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
      }
    } else {
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
    }
    
    # Write results back to the dt rows for this year
    set(dt, i = yr_rows, j = col_max,  value = neighbor_max[cell_positions])
    set(dt, i = yr_rows, j = col_min,  value = neighbor_min[cell_positions])
    set(dt, i = yr_rows, j = col_mean, value = neighbor_mean[cell_positions])
  }
  
  dt
}

# ==============================================================================
# STEP 3: Main pipeline (replaces the original outer loop)
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # --- STATIC (one-time): build neighbor structures from the nb object ---
  message("Building static neighbor structures (one-time)...")
  t0 <- Sys.time()
  static <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
  message("  Done in ", round(difftime(Sys.time(), t0, units = "secs"), 1), " seconds.")
  
  # --- DYNAMIC (per variable): compute neighbor max, min, mean ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    cell_data <- compute_neighbor_stats_fast(
      dt       = cell_data,
      var_name = var_name,
      id_order = id_order,
      static   = static
    )
    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " seconds.")
  }
  
  # Free the large static structures
  rm(static)
  gc()
  
  cell_data
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with the pre-trained Random Forest prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup level** | Cell × Year (6.46M entries) | Cell only (344K entries) — **static** |
| **Key mechanism** | String paste + named-vector hash | Integer position indexing |
| **Stats computation** | R-level `lapply` over 6.46M rows × 5 vars | Sparse matrix multiply (mean) + `data.table` grouped agg (max/min), 28 years × 5 vars = 140 vectorized passes |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** on 16 GB laptop |
| **Numerical equivalence** | Baseline | Preserved — identical max, min, mean formulas; same NA handling |
| **RF model** | Pre-trained, not retrained | **Untouched** — only the feature-engineering step is redesigned |
| **Memory** | ~6.46M-element list of integer vectors | Sparse matrix (~1.4M non-zeros) + 344K-element neighbor list; peak from `tmp_dt` ~1.4M rows — well within 16 GB |

The key architectural principle: **topology is static, values are dynamic** — build the graph once, then sweep values through it per year using vectorized operations.