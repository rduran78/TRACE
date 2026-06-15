 **Diagnosis**  
The bottleneck arises from:  
- `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`.  
- Repeated string concatenation (`paste`) and list indexing for every row.  
- Memory overhead from large lists and repeated subsetting.  

The current approach is not vectorized and scales poorly for millions of rows.  

---

**Optimization Strategy**  
1. **Precompute keys and join instead of per-row lookup**: Avoid building neighbor indices row-by-row. Instead, create a long-format mapping of `(cell_id, year)` to neighbors and merge.  
2. **Use `data.table` for fast joins and aggregation**: Handles large data efficiently in memory.  
3. **Compute neighbor stats in a single grouped operation** rather than per-row `lapply`.  
4. **Avoid repeated `paste` calls**: Store keys as integers or factors.  
5. **Parallelize aggregation** if possible (`data.table` supports multithreading).  

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table
cell_data_dt <- as.data.table(cell_data)

# Precompute neighbor relationships in long format
# rook_neighbors_unique: list of neighbors for each id in id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand for all years
years <- unique(cell_data_dt$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]

# Merge neighbor values for each source variable
setkey(cell_data_dt, id, year)
setkey(neighbor_dt, neighbor_id, year)

compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  joined <- neighbor_dt[cell_data_dt, on = .(neighbor_id = id, year), nomatch = 0]
  # Compute stats grouped by original id-year
  stats <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  # Merge back to cell_data_dt
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                    paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  cell_data_dt[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
for (var_name in neighbor_source_vars) {
  cell_data_dt <- compute_neighbor_stats_dt(var_name)
}

# Final optimized dataset
cell_data <- as.data.frame(cell_data_dt)
```

---

**Why This Works**  
- Eliminates per-row loops: uses vectorized joins and group aggregation.  
- `data.table` handles millions of rows efficiently with low memory overhead.  
- Preserves original estimand and Random Forest model (only feature engineering changes).  
- Expected runtime: **minutes to a few hours** instead of 86+ hours on a standard laptop.  

**Additional Tips**  
- Ensure `setDTthreads()` uses all cores.  
- If memory is still tight, process variables one at a time and write intermediate results to disk.  
- Consider compressing intermediate tables with `fst` or `arrow`.