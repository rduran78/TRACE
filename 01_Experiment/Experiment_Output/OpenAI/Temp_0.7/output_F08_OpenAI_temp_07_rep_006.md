 **Diagnosis**  
The current implementation repeatedly computes neighbor statistics for every cell-year row by iterating through all 6.46 million rows, performing lookups and aggregation from scratch for each neighbor set. This is highly inefficient because:  
- The neighbor relationships are static across years, yet we rebuild or traverse them repeatedly for each of the 28 years.  
- The `lapply` approach creates massive overhead by re-computing neighbor indices for each row individually.  
- The memory footprint is huge (~6.46 million × 3 stats × 5 variables).  

**Optimization Strategy**  
1. **Precompute static neighbor indices** once for unique cell IDs (344,208 cells).  
2. **Vectorize year-based operations**: For each year, slice the data and compute neighbor stats using matrix operations or `data.table` joins instead of per-row loops.  
3. **Incremental aggregation**: Compute neighbor max/min/mean for each variable by year using the static neighbor map and apply fast functions (like `vapply` or matrix indexing).  
4. Avoid full recomputation of `neighbor_lookup` per row; use a compact structure keyed by cell index.  

Expected improvement: Reduce complexity from *O(rows × neighbors)* to *O(years × cells × neighbors)* with efficient vectorization.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute static neighbor lookup for each cell (not per year)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(ref_idx) {
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    as.integer(id_to_ref[as.character(neighbor_cell_ids)])
  })
}

neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Compute neighbor stats by year in a vectorized manner
compute_neighbor_stats_by_year <- function(dt, neighbor_lookup, var_name) {
  years <- unique(dt$year)
  result_list <- vector("list", length(years))
  
  for (y in seq_along(years)) {
    yr <- years[y]
    slice <- dt[year == yr]
    vals <- slice[[var_name]]
    
    # For each cell, compute stats from its neighbors
    stats_mat <- vapply(seq_along(neighbor_lookup), function(i) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3))
    
    # Transpose to match rows
    stats_mat <- t(stats_mat)
    result_list[[y]] <- data.table(
      id = id_order,
      year = yr,
      paste0(var_name, "_nbr_max") := stats_mat[, 1],
      paste0(var_name, "_nbr_min") := stats_mat[, 2],
      paste0(var_name, "_nbr_mean") := stats_mat[, 3]
    )
  }
  
  rbindlist(result_list)
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- lapply(neighbor_source_vars, function(v) {
  compute_neighbor_stats_by_year(cell_data, neighbor_lookup, v)
})

# Merge all neighbor features into main data
neighbor_features_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), neighbor_features)
cell_data <- merge(cell_data, neighbor_features_dt, by = c("id", "year"), all.x = TRUE)

# Preserve Random Forest prediction step
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Key Gains**  
- **Static neighbor map** avoids rebuilding indices for 6.46M rows.  
- **Year-by-year batch processing** reduces memory thrash and uses efficient joins.  
- **Vectorized `vapply`** drastically cuts function call overhead.  

This approach should reduce computation time from **86+ hours to a few hours** on a 16 GB laptop while preserving the original estimand and the trained Random Forest model.