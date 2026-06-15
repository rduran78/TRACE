 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `lapply` over 6.46M rows creates huge R lists in memory.  
- `build_neighbor_lookup` repeats string concatenation and lookups for every row.  
- `compute_neighbor_stats` repeatedly subsets vectors and computes stats in R loops.  
- No vectorization or efficient join strategy is used.  

**Optimization Strategy**  
- Avoid per-row `lapply`. Use **data.table** for fast joins and grouping.  
- Precompute neighbor relationships as a long table: `(cell_id, year, neighbor_id)`.  
- Join neighbor values once and compute `max`, `min`, `mean` by `cell_id, year`.  
- Process variables in a loop but reuse the same neighbor mapping.  
- Use `fread`/`fwrite` or `setDT` for memory efficiency.  
- This approach reduces complexity from O(N × neighbors) in R loops to efficient grouped aggregation in C via data.table.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# id_order: vector of unique cell ids
# rook_neighbors_unique: list of integer vectors (spdep nb object)

# 1. Build neighbor mapping as a long table
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# 2. Expand to panel by joining with years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# 3. Prepare for joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# 4. Compute neighbor stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  setnames(tmp, var_name, "neighbor_val")
  
  # Aggregate stats by (id, year)
  stats <- tmp[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle all-NA cases
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added
```

---

**Why This Works**  
- Converts neighbor relationships into a **long table** and uses **data.table joins** and **grouped aggregation** in C, which is orders of magnitude faster than R loops.  
- Memory footprint is reduced because we avoid creating millions of small lists.  
- Preserves original estimand and trained model since only feature engineering changes.  

**Expected Performance**  
- From 86+ hours to **minutes or a few hours** on a 16 GB laptop, depending on disk I/O and CPU.  
- Scales well for 6.46M rows and 1.37M neighbor links.  

**Tip**: If memory is still tight, process one variable at a time (as shown) and write intermediate results to disk with `fwrite()`.