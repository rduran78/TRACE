 The evidence points to the **neighbor feature computation**, not Random Forest inference, as the main bottleneck.  

### **Diagnosis**
- The dataset has **6.46M rows**, and for each of 5 variables, `compute_neighbor_stats()` iterates over **all rows**, performing multiple `lapply()` calls and repeated vector slicing.
- `build_neighbor_lookup()` creates a large list of neighbor indices via `lapply()` over 6.46M rows, which is enormously expensive in both time and memory.
- Random Forest inference on 6.46M rows × 110 variables is typically minutes to an hour at most, whereas these nested loops and list operations scale poorly and explain the 86+ hour runtime.

### **Optimization Strategy**
- Precompute neighbor relationships at the **cell level** (344,208 cells) instead of cell-year level.
- Use **vectorized joins** with `data.table` instead of `lapply()`.
- Compute neighbor stats in bulk using group operations, avoiding per-row loops.
- Reuse `neighbor_lookup` efficiently without duplicating work for each year-variable combination.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Add a unique key for cell-year
cell_data[, cell_year := paste(id, year, sep = "_")]

# Build neighbor pairs at cell level (not per row)
id_order_dt <- data.table(id = id_order, idx = seq_along(id_order))
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand neighbor pairs across years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# Join with values for each variable and compute stats
for (var_name in neighbor_source_vars) {
  # Create neighbor value table
  neighbor_vals <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  # Join to get neighbor values
  joined <- merge(neighbor_pairs, neighbor_vals, by = c("neighbor_id", "year"), all.x = TRUE)
  # Compute stats per (id, year)
  stats <- joined[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# Clean up
cell_data[, cell_year := NULL]
```

---

### **Why This Works**
- Eliminates per-row `lapply()` over 6.46M rows; instead, uses **data.table joins and grouped aggregation**.
- Reduces complexity from O(N × neighbors × variables) in R loops to optimized C-level group operations.
- Handles all years in one pass and avoids reconstructing neighbor lookups repeatedly.
- Memory footprint stays manageable because neighbor pairs (~1.37M) × 28 years ≈ 38M rows, which is large but feasible with `data.table`.

**Expected runtime reduction:** From 86+ hours to a few minutes to an hour on a 16 GB machine.  
**RF inference remains unchanged and will run much faster than the original neighbor computation bottleneck.**