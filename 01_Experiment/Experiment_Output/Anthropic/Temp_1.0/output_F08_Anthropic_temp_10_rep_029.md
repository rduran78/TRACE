 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) relationships.** It creates a list of length ~6.46 million, where each element maps a cell-year row to its neighbor cell-year rows. But the neighbor *topology* is identical across all 28 years — only the variable *values* change. This means the function does **28× redundant work** discovering the same spatial relationships, and creates a lookup structure **28× larger than necessary**.

2. **String-key hashing is expensive.** The function pastes `id_year` strings and indexes into a named character vector (`idx_lookup`) for every single neighbor edge, across every year. With ~1.37M directed edges × 28 years, this produces ~38.5 million string-match lookups — all to rediscover the same cell→cell mapping that is already encoded in `rook_neighbors_unique`.

3. **`compute_neighbor_stats` operates on the bloated 6.46M-element lookup.** Each call iterates over all cell-year rows rather than operating on a per-year matrix slice, defeating cache locality and vectorization opportunities.

4. **Memory pressure.** The 6.46M-element list of integer vectors consumes substantial RAM and puts GC pressure on a 16 GB laptop.

### The Key Insight

> **Neighbor topology is static (cell-to-cell); only the attached variables are dynamic (year-varying).**

The neighbor list `rook_neighbors_unique` already defines, for each of the 344,208 cells, which other cells are its neighbors. This never changes. The only thing that changes across years is the *value* of variables like `ntl`, `ec`, etc. Therefore:

- Build the cell-to-cell neighbor index **once** (344K entries, not 6.46M).
- For each variable, reshape values into a **cells × years matrix**, then compute neighbor max/min/mean using vectorized operations on matrix columns (years), reusing the single static neighbor index.

This reduces complexity by a factor of ~28× in lookup construction and enables fully vectorized stats computation.

---

## Optimization Strategy

| Aspect | Current (Slow) | Redesigned (Fast) |
|---|---|---|
| Lookup granularity | cell×year (6.46M entries) | cell only (344K entries) |
| Lookup keys | String paste + named vector match | Direct integer indexing from `nb` object |
| Stats computation | `lapply` over 6.46M rows per variable | Vectorized matrix operations over 344K cells × 28 years |
| Total lookup builds | 1 (but huge) | 1 (small) |
| Stats calls | 5 × 6.46M = 32.3M `lapply` iterations | 5 variables × 28 years, each vectorized over 344K cells |
| Estimated speedup | Baseline (~86+ hrs) | **~20–60 minutes** (conservative) |

### Algorithmic Steps

1. **Build a static cell-level neighbor index** from `rook_neighbors_unique`: a list of 344K integer vectors mapping each cell's positional index to its neighbors' positional indices.

2. **Establish a stable cell ordering** so that cell `i` in the neighbor index corresponds to row `i` in a values matrix.

3. **Reshape each variable into a 344,208 × 28 matrix** (cells × years).

4. **For each variable, compute neighbor stats in vectorized fashion:**
   - For each cell, gather neighbor values from the matrix (all years at once via matrix row-subsetting).
   - Use optimized row-wise max/min/mean across neighbor sets.

5. **Write results back** into the original `cell_data` data frame in the same column format as before, preserving the exact numerical estimand for the pre-trained Random Forest.

---

## Working R Code

```r
# =============================================================================
# REDESIGNED NEIGHBOR STATS COMPUTATION
# Exploits: static topology (cell-to-cell) + dynamic values (year-varying)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, 
                                           neighbor_source_vars) {
  
  # -------------------------------------------------------------------------
  # STEP 0: Convert to data.table for speed (non-destructive)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # -------------------------------------------------------------------------
  # STEP 1: Build STATIC cell-level neighbor index (done ONCE)
  #
  # rook_neighbors_unique is an spdep::nb object: a list of length N where

  # element [[i]] contains integer indices of neighbors of cell i, referencing
  # positions in id_order. A value of 0L means no neighbors.
  # We convert this to a clean list of integer vectors.
  # -------------------------------------------------------------------------
  message("Building static cell-level neighbor index (", 
          format(length(id_order), big.mark = ","), " cells)...")
  
  n_cells <- length(id_order)
  
  # spdep::nb stores 0L for no-neighbor cells; clean those out
  static_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb != 0L]
    as.integer(nb)
  })
  # Now static_neighbor_idx[[i]] gives the positional indices (in id_order)
  # of the neighbors of the i-th cell in id_order.
  
  message("  Static neighbor index built. Length: ", length(static_neighbor_idx))
  
  # -------------------------------------------------------------------------
  # STEP 2: Establish stable cell ordering and year ordering
  #
  # We need a mapping: cell id -> position in id_order (1..N)
  # And: year -> column index in our values matrix
  # -------------------------------------------------------------------------
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  message("  Cells: ", format(n_cells, big.mark = ","), 
          ", Years: ", n_years, " (", min(years), "-", max(years), ")")
  
  # -------------------------------------------------------------------------
  # STEP 3: Map each row in dt to (cell_position, year_column)
  # -------------------------------------------------------------------------
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]
  
  # Verify complete mapping
  stopifnot(!anyNA(dt$cell_pos), !anyNA(dt$year_col))
  
  # Create a row-index matrix: row_index_mat[cell_pos, year_col] = row in dt
  # This lets us write results back to the correct rows.
  row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_index_mat[cbind(dt$cell_pos, dt$year_col)] <- seq_len(nrow(dt))
  
  # -------------------------------------------------------------------------
  # STEP 4: For each variable, reshape → compute neighbor stats → write back
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    message("Processing neighbor stats for: ", var_name, " ...")
    t_start <- proc.time()
    
    # 4a. Reshape variable into cells × years matrix
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]
    
    # 4b. Compute neighbor max, min, mean for all cells × all years
    #     Result matrices: same dimension as val_mat
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Pre-compute: for each cell, extract the neighbor rows from val_mat
    # This is the inner loop over 344K cells (vectorized across years)
    for (i in seq_len(n_cells)) {
      nb_idx <- static_neighbor_idx[[i]]
      if (length(nb_idx) == 0L) next
      
      # nb_vals is a (num_neighbors × n_years) matrix
      # Each column is one year; each row is one neighbor's value that year
      if (length(nb_idx) == 1L) {
        # Single neighbor: matrix with 1 row
        nb_vals <- matrix(val_mat[nb_idx, ], nrow = 1L, ncol = n_years)
      } else {
        nb_vals <- val_mat[nb_idx, , drop = FALSE]
      }
      
      # Vectorized column-wise (i.e., per-year) stats using colMins etc.
      # Using matrixStats if available, otherwise base R
      # For robustness, use base R apply or manual colwise ops:
      nb_max[i, ]  <- apply(nb_vals, 2L, max,  na.rm = TRUE)
      nb_min[i, ]  <- apply(nb_vals, 2L, min,  na.rm = TRUE)
      nb_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
    
    # Handle -Inf/Inf from max/min on all-NA columns → convert to NA
    nb_max[is.infinite(nb_max)]  <- NA_real_
    nb_min[is.infinite(nb_min)]  <- NA_real_
    nb_mean[is.nan(nb_mean)]     <- NA_real_
    
    # 4c. Write results back to data.table using the row index mapping
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Flatten matrices back to vector aligned with dt rows
    valid <- !is.na(row_index_mat)
    dt_rows   <- row_index_mat[valid]
    max_vals  <- nb_max[valid]
    min_vals  <- nb_min[valid]
    mean_vals <- nb_mean[valid]
    
    set(dt, i = dt_rows, j = max_col,  value = max_vals)
    set(dt, i = dt_rows, j = min_col,  value = min_vals)
    set(dt, i = dt_rows, j = mean_col, value = mean_vals)
    
    elapsed <- (proc.time() - t_start)["elapsed"]
    message("  Done: ", var_name, " in ", round(elapsed / 60, 1), " minutes.")
  }
  
  # -------------------------------------------------------------------------
  # STEP 5: Clean up helper columns, return as data.frame
  # -------------------------------------------------------------------------
  dt[, c("cell_pos", "year_col") := NULL]
  
  message("All neighbor features computed.")
  return(as.data.frame(dt))
}


# =============================================================================
# OPTIONAL: Faster version using matrixStats (if installed)
# Replaces the inner apply() calls with compiled C routines
# =============================================================================

compute_all_neighbor_features_fast <- function(cell_data, id_order, 
                                                rook_neighbors_unique, 
                                                neighbor_source_vars) {
  
  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    message("matrixStats not found; falling back to base R version.")
    return(compute_all_neighbor_features(cell_data, id_order, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars))
  }
  
  library(data.table)
  library(matrixStats)
  
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)
  
  # Static neighbor index
  static_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb != 0L]
    as.integer(nb)
  })
  
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]
  
  row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_index_mat[cbind(dt$cell_pos, dt$year_col)] <- seq_len(nrow(dt))
  
  # Pre-compute CSR-like structure for batch processing
  message("Building CSR neighbor structure for vectorized access...")
  nb_lengths <- vapply(static_neighbor_idx, length, integer(1L))
  nb_flat    <- unlist(static_neighbor_idx, use.names = FALSE)
  nb_offsets <- c(0L, cumsum(nb_lengths))
  # nb_offsets[i]+1 .. nb_offsets[i+1] gives positions in nb_flat for cell i
  
  for (var_name in neighbor_source_vars) {
    
    message("Processing: ", var_name, " ...")
    t_start <- proc.time()
    
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]
    
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Process in chunks to manage memory and allow progress reporting
    chunk_size <- 10000L
    n_chunks <- ceiling(n_cells / chunk_size)
    
    for (ch in seq_len(n_chunks)) {
      i_start <- (ch - 1L) * chunk_size + 1L
      i_end   <- min(ch * chunk_size, n_cells)
      
      for (i in i_start:i_end) {
        n_nb <- nb_lengths[i]
        if (n_nb == 0L) next
        
        from <- nb_offsets[i] + 1L
        to   <- nb_offsets[i + 1L]
        nb_idx <- nb_flat[from:to]
        
        if (n_nb == 1L) {
          # Single neighbor: just copy that row
          nb_max[i, ]  <- val_mat[nb_idx, ]
          nb_min[i, ]  <- val_mat[nb_idx, ]
          nb_mean[i, ] <- val_mat[nb_idx, ]
        } else {
          nb_vals <- val_mat[nb_idx, , drop = FALSE]
          nb_max[i, ]  <- colMaxs(nb_vals,  na.rm = TRUE)
          nb_min[i, ]  <- colMins(nb_vals,  na.rm = TRUE)
          nb_mean[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
        }
      }
      
      if (ch %% 5 == 0 || ch == n_chunks) {
        message("  Chunk ", ch, "/", n_chunks, " done.")
      }
    }
    
    nb_max[is.infinite(nb_max)]  <- NA_real_
    nb_min[is.infinite(nb_min)]  <- NA_real_
    nb_mean[is.nan(nb_mean)]     <- NA_real_
    
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    valid     <- !is.na(row_index_mat)
    dt_rows   <- row_index_mat[valid]
    set(dt, i = dt_rows, j = max_col,  value = nb_max[valid])
    set(dt, i = dt_rows, j = min_col,  value = nb_min[valid])
    set(dt, i = dt_rows, j = mean_col, value = nb_mean[valid])
    
    elapsed <- (proc.time() - t_start)["elapsed"]
    message("  ", var_name, ": ", round(elapsed / 60, 1), " min")
  }
  
  dt[, c("cell_pos", "year_col") := NULL]
  return(as.data.frame(dt))
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Replace the original 3-step process:
#   neighbor_lookup <- build_neighbor_lookup(...)
#   for (var_name in ...) { cell_data <- compute_and_add_neighbor_features(...) }
#
# With this single call:
cell_data <- compute_all_neighbor_features_fast(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# Then proceed to prediction with the pre-trained Random Forest as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` computed on exactly the same neighbor sets with `na.rm = TRUE`, producing identical values to the original. The `is.infinite` → `NA` and `is.nan` → `NA` cleanup matches the original's `return(c(NA, NA, NA))` for all-NA or empty neighbor cases. |
| **Column naming** | Produces `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` — same as the original `compute_and_add_neighbor_features`. |
| **Pre-trained RF preserved** | No model retraining. The function only computes features; prediction proceeds unchanged afterward. |
| **Memory on 16 GB laptop** | Peak: one `344,208 × 28` matrix (~77 MB) for values + three for results (~231 MB) + the data.table (~5-6 GB) + static neighbor index (~50 MB). Well within 16 GB. |
| **Expected runtime** | ~15–45 minutes total (vs. 86+ hours), depending on disk I/O and GC. The inner loop iterates 344K cells (not 6.46M rows), and each iteration is vectorized across 28 years via `colMaxs`/`colMins`/`colMeans2`. |