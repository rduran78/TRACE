 **Diagnosis**  
The current implementation is slow and memory-intensive because:  
- `lapply` over 6.46M rows creates large intermediate lists.  
- Repeated string concatenation (`paste`) and lookups are costly.  
- Neighbor stats are computed row-wise instead of vectorized.  
- No parallelization or efficient data structures (e.g., `data.table`) are used.  

**Optimization Strategy**  
1. **Switch to `data.table`** for fast joins and grouping.  
2. **Precompute neighbor relationships as a long table** instead of per-row lists.  
3. **Aggregate neighbor stats using vectorized operations** (`max`, `min`, `mean`) via `data.table` grouping.  
4. **Avoid repeated string operations**; use integer keys.  
5. **Parallelize** if possible using `future.apply` or `data.table` multithreading.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor relationships as a long table
# id_order and rook_neighbors_unique assumed available
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[, .(id, neighbor_id)][, .(year = years), by = .(id, neighbor_id)]

# Join neighbor_panel with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_panel, neighbor_id, year)

# For each source variable, compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join to get neighbor values
  neighbor_panel[, val := cell_data[.SD, on = .(id = neighbor_id, year), get(var_name)]]
  
  # Aggregate stats by (id, year)
  stats_dt <- neighbor_panel[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats_dt, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

# cell_data now contains neighbor features efficiently
```

**Why This Works**  
- Converts neighbor lookups into a long table, avoiding per-row list overhead.  
- Uses `data.table` joins and grouping for fast aggregation.  
- Eliminates repeated string concatenation and large `lapply` calls.  
- Scales well on 6.46M rows and leverages in-memory efficiency.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop, depending on disk I/O and CPU cores.