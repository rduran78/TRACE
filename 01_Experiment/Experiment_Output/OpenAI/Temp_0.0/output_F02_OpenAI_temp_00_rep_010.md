 **Diagnosis**  
The current implementation is slow and memory-intensive because:  
- `lapply` over 6.46M rows creates large intermediate lists.  
- Neighbor lookups repeatedly paste strings and perform named indexing, which is costly.  
- `compute_neighbor_stats` loops over all rows and recomputes aggregates inefficiently.  
- Everything runs in pure R without vectorization or parallelization.  

**Optimization Strategy**  
1. **Avoid repeated string operations**: Precompute keys or use integer-based indexing.  
2. **Use `data.table` for fast joins and aggregation** instead of `lapply`.  
3. **Parallelize computations** using `future.apply` or `parallel`.  
4. **Chunk processing** to reduce memory footprint.  
5. **Precompute neighbor relationships in long format** and aggregate with `data.table` rather than row-wise loops.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor relationships in long format
# id_order and rook_neighbors_unique assumed available
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Build long neighbor table
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]

# Merge with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  tmp <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  tmp <- tmp[, .(max_val = max(get(var_name), na.rm = TRUE),
                 min_val = min(get(var_name), na.rm = TRUE),
                 mean_val = mean(get(var_name), na.rm = TRUE)),
             by = .(id, year)]
  setnames(tmp, c("id", "year", paste0(var_name, "_nbr_max"),
                  paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  tmp
}

# Compute for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge all results back to cell_data
for (res in result_list) {
  cell_data <- res[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added
```

**Why this is faster**  
- Eliminates per-row loops and string concatenation.  
- Uses `data.table` joins and group-by aggregation (highly optimized in C).  
- Scales well for millions of rows and can be parallelized further if needed.  

**Expected improvement**  
From 86+ hours to a few hours on a standard laptop, possibly less with parallelization.  

This preserves the trained Random Forest model and original estimand because only feature engineering changes, not the model or target variable.