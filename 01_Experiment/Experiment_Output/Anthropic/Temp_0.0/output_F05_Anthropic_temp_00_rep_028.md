 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, repeated 6.46M times.

The string-key lookup on a named vector of length 6.46M is approximately O(N) per probe (R's named vector lookup is hash-based, but the hash table was built over 6.46M entries and is probed ~25.8M times total). The real cost is the **per-row R-level loop overhead** and **millions of small string allocations**.

### The Broader Pattern

The same neighbor topology is **identical across all 28 years** — cell `i`'s rook neighbors don't change over time. Yet the code re-discovers the spatial relationship for every cell-year row. This is the fundamental redundancy: **spatial structure is conflated with the panel (space × time) structure**.

Then `compute_neighbor_stats` is called 5 times, each time iterating over the 6.46M-element `neighbor_lookup` list. This is acceptable if the lookup is precomputed, but the lookup itself is the bottleneck.

## Optimization Strategy

**Principle: Separate space from time. Vectorize everything.**

1. **Build the neighbor lookup in spatial-ID space only** (344K cells, not 6.46M cell-years). Each cell has a fixed set of neighbor cell IDs. This is a one-time operation on 344K entries.

2. **For each variable and each year, use vectorized matrix indexing** to gather neighbor values and compute stats. No per-row R loop. No string keys.

3. **Use `data.table` for fast split-apply-combine** by year, with integer indexing into a spatial-ID lookup.

This reduces the algorithmic complexity from ~6.46M × (string ops + hash probes) to ~28 × (vectorized operations on 344K cells).

**Expected speedup**: From 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build spatial-only neighbor lookup (once, 344K cells)
# =============================================================================
# rook_neighbors_unique: spdep nb object, indexed by position in id_order
# id_order: vector of 344,208 cell IDs in the order matching the nb object

build_spatial_neighbor_lookup <- function(id_order, nb_obj) {
  # Returns a list of length length(id_order).
  # Element i contains the integer positions (in id_order) of cell i's neighbors.
  # This is essentially what the nb object already is, but we make it explicit.
  #
  # spdep nb objects already store neighbor indices as integer vectors
  # referencing positions in the original spatial object (= id_order here).
  # We just need to ensure no zero-length entries cause issues.
  
  n <- length(id_order)
  stopifnot(length(nb_obj) == n)
  
  # nb objects store integer indices; 0L means no neighbors in spdep convention
  lapply(nb_obj, function(x) {
    x <- as.integer(x)
    x[x != 0L]
  })
}

spatial_nb <- build_spatial_neighbor_lookup(id_order, rook_neighbors_unique)

# =============================================================================
# STEP 2: Convert to data.table and create spatial index
# =============================================================================
dt <- as.data.table(cell_data)

# Create a mapping from cell ID -> position in id_order (spatial index)
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to data
dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Verify
stopifnot(!anyNA(dt$spatial_idx))

# =============================================================================
# STEP 3: Vectorized neighbor stats computation
# =============================================================================
compute_all_neighbor_features <- function(dt, id_order, spatial_nb, var_names) {
  # Strategy:
  # For each year, we have up to 344,208 cells.
  # We build a value vector indexed by spatial position, then use the
  # neighbor list to gather values and compute stats — all vectorized.
  
  n_spatial <- length(id_order)
  years <- sort(unique(dt$year))
  
  # Pre-allocate result columns
  for (var_name in var_names) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  # Precompute neighbor lengths and a flat (row, col) structure for matrix ops
  # For each spatial cell, we know its neighbors. We build CSR-like vectors:
  nb_lengths <- vapply(spatial_nb, length, integer(1))  # length 344,208
  max_nb     <- max(nb_lengths)
  
  # Build a padded neighbor matrix: n_spatial x max_nb
  # Pad with NA so we can do matrix indexing
  nb_matrix <- matrix(NA_integer_, nrow = n_spatial, ncol = max_nb)
  for (i in seq_len(n_spatial)) {
    nbs <- spatial_nb[[i]]
    if (length(nbs) > 0L) {
      nb_matrix[i, seq_along(nbs)] <- nbs
    }
  }
  # nb_matrix[i, j] = spatial index of the j-th neighbor of cell i, or NA
  
  # For each year, fill a spatial-indexed value vector, then compute stats
  setkey(dt, year, spatial_idx)
  
  for (yr in years) {
    # Subset rows for this year
    yr_rows <- dt[.(yr)]  # keyed lookup
    
    # Build spatial_idx -> row index in yr_rows
    # (not all 344K cells may be present every year)
    yr_spatial_idx <- yr_rows$spatial_idx
    
    # Value vector indexed by spatial position (NA for missing cells)
    for (var_name in var_names) {
      val_vec <- rep(NA_real_, n_spatial)
      val_vec[yr_spatial_idx] <- yr_rows[[var_name]]
      
      # Gather neighbor values using the padded neighbor matrix
      # nb_matrix is n_spatial x max_nb; index into val_vec
      neighbor_vals_mat <- matrix(val_vec[nb_matrix], 
                                  nrow = n_spatial, ncol = max_nb)
      # neighbor_vals_mat[i, j] = value of j-th neighbor of cell i, or NA
      
      # Compute row-wise stats (only for cells present this year)
      # Use matrixStats if available for speed, otherwise base R
      present <- yr_spatial_idx  # spatial indices of cells present this year
      
      sub_mat <- neighbor_vals_mat[present, , drop = FALSE]
      
      # rowMins, rowMaxs, rowMeans ignoring NA
      # Base R approach (no extra dependency):
      row_max  <- apply(sub_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
      })
      row_min  <- apply(sub_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
      })
      row_mean <- apply(sub_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x)
      })
      
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Write back into dt for this year's rows
      # We need the actual row indices in dt
      dt_row_idx <- which(dt$year == yr)
      # These are in the same order as yr_spatial_idx because of the key
      dt[dt_row_idx, (col_max)  := row_max]
      dt[dt_row_idx, (col_min)  := row_min]
      dt[dt_row_idx, (col_mean) := row_mean]
    }
    
    message(sprintf("Year %d complete.", yr))
  }
  
  return(dt)
}

# =============================================================================
# STEP 3b: Even faster version using matrixStats (recommended)
# =============================================================================
compute_all_neighbor_features_fast <- function(dt, id_order, spatial_nb, var_names) {
  
  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    stop("Install matrixStats for the fast path: install.packages('matrixStats')")
  }
  
  n_spatial <- length(id_order)
  years <- sort(unique(dt$year))
  
  # Pre-allocate result columns
  for (var_name in var_names) {
    dt[, paste0("neighbor_max_", var_name)  := NA_real_]
    dt[, paste0("neighbor_min_", var_name)  := NA_real_]
    dt[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  # Build padded neighbor matrix (one-time cost)
  nb_lengths <- vapply(spatial_nb, length, integer(1))
  max_nb <- max(nb_lengths)
  
  nb_matrix <- matrix(NA_integer_, nrow = n_spatial, ncol = max_nb)
  for (i in seq_len(n_spatial)) {
    nbs <- spatial_nb[[i]]
    if (length(nbs) > 0L) {
      nb_matrix[i, seq_along(nbs)] <- nbs
    }
  }
  
  # Create a row-index column for fast assignment
  dt[, .row_id := .I]
  setkey(dt, year)
  
  for (yr in years) {
    # Get row indices in dt for this year
    dt_idx <- dt[.(yr), which = TRUE]
    yr_spatial <- dt$spatial_idx[dt_idx]
    
    for (var_name in var_names) {
      # Build spatial value vector
      val_vec <- rep(NA_real_, n_spatial)
      val_vec[yr_spatial] <- dt[[var_name]][dt_idx]
      
      # Gather neighbor values: only for present cells
      sub_nb <- nb_matrix[yr_spatial, , drop = FALSE]
      sub_vals <- matrix(val_vec[sub_nb], nrow = length(yr_spatial), ncol = max_nb)
      
      # matrixStats handles NA natively and is C-optimized
      r_max  <- matrixStats::rowMaxs(sub_vals,  na.rm = TRUE)
      r_min  <- matrixStats::rowMins(sub_vals,  na.rm = TRUE)
      r_mean <- matrixStats::rowMeans2(sub_vals, na.rm = TRUE)
      
      # matrixStats returns -Inf/Inf/NaN when all NA; fix to NA
      all_na <- matrixStats::rowAlls(is.na(sub_vals))
      r_max[all_na]  <- NA_real_
      r_min[all_na]  <- NA_real_
      r_mean[all_na] <- NA_real_
      
      set(dt, i = dt_idx, j = paste0("neighbor_max_", var_name),  value = r_max)
      set(dt, i = dt_idx, j = paste0("neighbor_min_", var_name),  value = r_min)
      set(dt, i = dt_idx, j = paste0("neighbor_mean_", var_name), value = r_mean)
    }
    
    message(sprintf("Year %d complete.", yr))
  }
  
  dt[, .row_id := NULL]
  return(dt)
}

# =============================================================================
# STEP 4: Execute
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data_dt <- compute_all_neighbor_features_fast(
  dt          = as.data.table(cell_data),
  id_order    = id_order,
  spatial_nb  = spatial_nb,
  var_names   = neighbor_source_vars
)

# Convert back to data.frame if needed downstream
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is untouched — 
# only the feature-construction pipeline has been optimized.
# The numerical estimand is preserved: same max, min, mean over 
# the same rook-neighbor sets per cell-year.
```

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **Loop iterations** | 6.46M (R-level `lapply`) | 28 years × 5 vars = 140 vectorized passes |
| **String allocations** | ~25.8M `paste()` calls + 6.46M named-vector probes | **Zero** |
| **Lookup structure** | 6.46M-element named character vector | Integer matrix indexing |
| **Neighbor discovery** | Per cell-year row (redundant across years) | Per cell (spatial only, once) |
| **Stats computation** | 5 × 6.46M list iterations | 140 × vectorized `matrixStats` on ~344K rows |
| **Memory** | 6.46M-element list of integer vectors | One `344K × max_nb` integer matrix (~5–11 MB) |
| **Estimated time** | 86+ hours | **~5–15 minutes** |

## Summary

The string-key construction was the visible symptom, but the root cause was **treating a space × time panel as flat and re-discovering spatial topology per row via string hashing**. The fix separates the spatial neighbor structure (which is time-invariant) from the temporal panel, builds a padded integer neighbor matrix once, and uses vectorized matrix indexing + `matrixStats` to compute all neighbor statistics in bulk per year. The trained Random Forest model and the numerical estimand (max, min, mean of rook-neighbor values per cell-year) are exactly preserved.