 **Diagnosis**  
The current bottlenecks come from:  
- Using `lapply` over 6.46M rows for neighbor lookups and neighbor statistics creates huge R object overhead and memory pressure.  
- `build_neighbor_lookup` constructs a large list of integer vectors by looping through each row, repeating year concatenation unnecessarily.  
- `compute_neighbor_stats` iterates over each row again, duplicating expensive subset operations.  
- No vectorized or table-based join approach; everything is in memory at the cell-year level.  

**Optimization Strategy**  
1. **Avoid row-wise loops:** Instead of storing per-row neighbor indices as a list, expand the neighbor relationships to a long table keyed by `year`. This allows joining and aggregating in a vectorized manner.  
2. **Compute all neighbor stats in a single grouped operation** using `data.table` for speed and memory efficiency.  
3. Do not duplicate neighbor lookups by year repeatedly; build a single `long` mapping between `(cell_id, year) → neighbors` and aggregate.  
4. Process variables one by one but reuse the neighbor long table to reduce RAM.  

---

### **Optimized Workflow**

**Step 1: Prepare data.table and neighbor map**
```r
library(data.table)

setDT(cell_data)  # Assuming columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Convert id_order to mapping
id_order_dt <- data.table(id = id_order, ref = seq_along(id_order))

# Build neighbor map: each cell id and its neighbors
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id    = id_order[i],
    neigh = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Repeat neighbor relationships across years
years <- unique(cell_data$year)
neighbor_long <- neighbor_pairs[ , .(year = years), by = .(id, neigh)]
```

**Step 2: Reshape long panel data and join**
```r
# Merge neighbor vars: key by (id, year) for center cell
setkey(cell_data, id, year)

# Example: compute neighbor stats for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor values from cell_data
  nb_dt <- merge(neighbor_long, cell_data[, .(neigh = id, year, val = get(var))],
                 by = c("neigh", "year"), all.x = TRUE)
  
  # Compute stats grouped by cell and year
  stats_dt <- nb_dt[!is.na(val),
    .(max_val = max(val), min_val = min(val), mean_val = mean(val)),
    by = .(id, year)]
  
  # Merge back into main data
  setnames(stats_dt, c("id", "year", paste0(var, "_nmax"), paste0(var, "_nmin"), paste0(var, "_nmean")))
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
}
```

---

### **Why This Works**
- Eliminates per-row `lapply`; uses `data.table` grouping and joins.
- Handles **6.46M rows** efficiently with vectorized aggregation.
- Memory-friendly: processes one variable at a time, minimal intermediate lists.
- Reuses a single neighbor-year map; no expensive string pasting.
- Preserves the existing model and numerical estimands.

**Expected Performance**: On a modern laptop with 16 GB RAM, runtime should reduce from **86+ hours to under a few hours**, depending on disk speed.  

**Tip**: Save intermediate `neighbor_long` and reuse in future runs. Additionally, if RAM is still an issue, process by year batches (e.g., 5 years at a time).