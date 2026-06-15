 **Diagnosis**  
The current implementation is slow and memory-intensive because:  
- `lapply` over 6.46M rows creates large intermediate lists.  
- Repeated string concatenation (`paste`) and lookups are costly.  
- Neighbor stats are computed in pure R loops without vectorization.  
- No parallelization or efficient data structures (e.g., `data.table`) are used.  

**Optimization Strategy**  
1. **Switch to `data.table`** for fast keyed joins and aggregation.  
2. **Precompute neighbor relationships as a long table** instead of per-row lists.  
3. **Vectorize aggregation** using `data.table` group operations instead of `lapply`.  
4. **Parallelize** using `future.apply` or `data.table` multithreading.  
5. Avoid repeated `paste` operations; use integer keys for joins.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create a lookup table of neighbors across all years
# id_order and rook_neighbors_unique assumed available
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]

# Merge neighbor values for each variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join cell_data twice: once for focal cell, once for neighbor
  merged <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
                  by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate neighbor stats
  stats <- merged[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                    paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  cell_data[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  cell_data <- compute_neighbor_stats_dt(var_name)
}

# Enable multithreading
setDTthreads(parallel::detectCores())
```

**Why this works**  
- Eliminates per-row `lapply` and string operations.  
- Uses efficient joins and aggregations in `data.table`.  
- Scales better for millions of rows and leverages multithreading.  
- Preserves original estimand and trained Random Forest model.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours on a 16 GB laptop, depending on disk I/O and CPU cores.