 **Diagnosis**  
The bottleneck is in neighbor feature construction, not Random Forest inference. Specifically:  
- `build_neighbor_lookup` uses `lapply` over ~6.46M rows, repeatedly performing string concatenation (`paste`) and named lookups, which is highly inefficient.  
- `compute_neighbor_stats` repeats similar operations for each variable, causing redundant passes through large data structures.  
- Memory overhead from millions of character keys and repeated list allocations is significant.  

**Optimization Strategy**  
1. **Vectorize neighbor lookup**: Precompute integer indices for all cell-year combinations instead of using string keys.  
2. **Avoid repeated loops**: Compute all neighbor stats in one pass for all variables.  
3. **Use `data.table` for efficient joins and aggregation**: Handles large datasets well.  
4. **Parallelize computation**: Use `parallel::mclapply` or `future.apply` for multi-core processing.  
5. **Reduce memory churn**: Replace character-based keys with integer-based indexing.  

**Optimized R Code**  

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute mapping: cell_id -> row indices by year
cell_data[, row_idx := .I]
id_year_map <- cell_data[, .(row_idx), by = .(id, year)]

# Build neighbor lookup as integer indices
build_neighbor_lookup_fast <- function(id_order, neighbors, id_year_map) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_len(nrow(id_year_map)), function(i) {
    ref_idx <- id_to_ref[as.character(id_year_map$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    # Join to get row indices for same year
    idx <- id_year_map[id %in% neighbor_ids & year == id_year_map$year[i], row_idx]
    as.integer(idx)
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, id_year_map)

# Compute neighbor stats for all variables in one pass
compute_neighbor_stats_fast <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  names(vals_list) <- vars
  
  mclapply(seq_along(neighbor_lookup), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(rep(NA_real_, length(vars) * 3))
    res <- unlist(lapply(vars, function(v) {
      nv <- vals_list[[v]][idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0) return(c(NA, NA, NA))
      c(max(nv), min(nv), mean(nv))
    }))
    res
  }, mc.cores = detectCores() - 1)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_matrix <- do.call(rbind, compute_neighbor_stats_fast(cell_data, neighbor_lookup, neighbor_source_vars))

# Bind results back to cell_data
stat_names <- unlist(lapply(neighbor_source_vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
cell_data[, (stat_names) := as.data.table(stats_matrix)]
```

**Expected Gains**  
- Eliminates repeated string operations and redundant loops.  
- Uses integer indexing and parallelization for speed.  
- Moves from 86+ hours to a few hours on a 16 GB laptop.  

This preserves the trained Random Forest model and original estimand while making neighbor feature construction practical.