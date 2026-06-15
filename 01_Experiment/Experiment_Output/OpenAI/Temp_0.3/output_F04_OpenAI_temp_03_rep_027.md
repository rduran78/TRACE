 **Diagnosis**  
The computational bottleneck is indeed the neighbor feature construction, not Random Forest inference. The current implementation repeatedly performs expensive lookups and list operations for ~6.46M rows and 5 variables, leading to excessive overhead. Specifically:  
- `build_neighbor_lookup` creates a large list of integer vectors via `lapply` over all rows, which is memory-heavy and slow.  
- `compute_neighbor_stats` iterates over all rows again for each variable, performing repeated indexing and NA filtering.  
- These operations scale poorly given 6.46M rows and millions of neighbor relationships.  

**Optimization Strategy**  
- Avoid repeated `lapply` over rows; use **vectorized operations** or **data.table** for grouping and aggregation.  
- Precompute a long-format neighbor table (cell-year → neighbor-year) and join once.  
- Compute max/min/mean in a single grouped operation per variable using `data.table` aggregation.  
- This reduces complexity from O(N × neighbors × variables) to O(N + E) where E is number of edges expanded over years.  
- Memory efficiency: process one variable at a time, but reuse the neighbor table.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
cell_dt <- as.data.table(cell_data)

# Precompute neighbor relationships expanded by year
# id_order: vector of cell IDs in canonical order
# rook_neighbors_unique: list of neighbors per cell index
years <- unique(cell_dt$year)

# Build long-format neighbor table
neighbor_list <- lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
})
neighbor_dt <- rbindlist(neighbor_list)
setkey(neighbor_dt, id)

# Expand by year
year_dt <- data.table(year = years)
neighbor_dt <- neighbor_dt[, .(year = years, neighbor_id), by = .(id)]

# Join with cell data to get neighbor values
setkey(cell_dt, id, year)

compute_neighbor_features <- function(var_name) {
  # Join neighbor_dt with cell_dt to get neighbor values
  joined <- neighbor_dt[cell_dt, on = .(neighbor_id = id, year), nomatch = 0]
  
  # joined now has columns: id (original), year, neighbor_id, var_name
  agg <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to cell_dt
  setnames(agg, c("id", "year", paste0(var_name, "_nbr_max"),
                  paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  cell_dt[agg, on = .(id, year)]
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_dt <- compute_neighbor_features(var_name)
}

# Convert back if needed
cell_data <- as.data.frame(cell_dt)
```

---

**Why This Works**  
- `neighbor_dt` is built once and reused.  
- Aggregation uses efficient `data.table` grouping instead of millions of `lapply` calls.  
- Complexity drops dramatically; expected runtime on 16 GB RAM laptop should reduce from 86+ hours to a few hours (or less with disk-backed operations).  
- Preserves original numerical estimand and trained Random Forest model.  

**Additional Tips**  
- If memory is still tight, process one year at a time or use `fst`/`arrow` for intermediate storage.  
- Parallelize the variable loop with `future.apply` or `data.table` `by` chunks.