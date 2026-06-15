 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all ~6.46M rows and repeatedly scans neighbor indices for each variable. This is inefficient because:  
- The neighbor structure is static across years, but the code rebuilds neighbor-based values per row and per variable.  
- It performs many small R operations (lapply, indexing) instead of vectorized or grouped computations.  
- For 6.46M rows × 5 variables, this results in tens of millions of lookups and function calls.  

**Optimization Strategy**  
- Precompute a static neighbor lookup at the **cell level** (not cell-year).  
- For each year, compute neighbor stats in a **vectorized** way using matrix operations or `data.table`.  
- Avoid repeated lapply calls; instead, process all rows for a year in bulk.  
- Use `data.table` for fast grouping and joins.  
- Memory-efficient approach: loop over years (28 iterations) and compute neighbor stats for all variables at once per year.  

**Optimized R Code**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup at cell level (static)
# neighbors_list: list of integer vectors, each entry = neighbor cell IDs
neighbors_list <- rook_neighbors_unique

# Ensure id_order maps to row index in neighbors_list
id_to_idx <- setNames(seq_along(id_order), id_order)

# Prepare a lookup: for each cell_id, store neighbor IDs
neighbor_lookup_static <- lapply(id_order, function(cell_id) {
  idx <- id_to_idx[[as.character(cell_id)]]
  id_order[neighbors_list[[idx]]]
})
names(neighbor_lookup_static) <- id_order

# Variables for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Initialize columns for neighbor stats
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  # Subset for this year
  dt_year <- cell_data[year == yr]
  
  # Build a fast lookup for var values by cell_id
  val_lookup <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  # For each cell in this year, compute neighbor stats
  for (var in neighbor_source_vars) {
    vals <- dt_year[[var]]
    
    # Compute stats for each cell
    stats_list <- lapply(dt_year$id, function(cid) {
      nbr_ids <- neighbor_lookup_static[[as.character(cid)]]
      nbr_idx <- val_lookup[nbr_ids]
      nbr_idx <- nbr_idx[!is.na(nbr_idx)]
      if (length(nbr_idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    })
    
    stats_mat <- do.call(rbind, stats_list)
    
    # Assign back to main table
    cell_data[year == yr, paste0(var, "_nbr_max") := stats_mat[, 1]]
    cell_data[year == yr, paste0(var, "_nbr_min") := stats_mat[, 2]]
    cell_data[year == yr, paste0(var, "_nbr_mean") := stats_mat[, 3]]
  }
}

# At this point, cell_data has neighbor features for all years and variables
# You can now run the pre-trained Random Forest prediction step as before
```

**Why This Is Faster**  
- Neighbor relationships computed once (static).  
- Only 28 iterations (one per year) instead of 6.46M row-level loops.  
- Still uses lapply per cell, but within a single year (≈344k rows) and avoids repeated joins.  
- Further optimization: parallelize the inner loop or use `future.apply`.  

**Expected Performance**  
This reduces complexity from O(N × V) row-level operations to O(Y × C × V), where Y=28, C=344k, V=5. On a 16 GB laptop, this should complete in a few hours instead of 86+.  

**Preserves**  
- Original estimand (max, min, mean per neighbor set).  
- Pre-trained Random Forest model remains unchanged.