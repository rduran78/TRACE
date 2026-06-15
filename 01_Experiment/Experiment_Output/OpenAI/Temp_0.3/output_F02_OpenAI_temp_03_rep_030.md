 **Diagnosis**  
The bottleneck is the heavy use of `lapply` over ~6.46M rows, repeatedly building neighbor-based aggregates in R lists. Each iteration allocates memory and performs lookups in R’s interpreter, which is extremely slow and memory-intensive for this scale. The current approach is essentially O(N × K) in pure R loops, where N ≈ 6.46M and K ≈ average neighbors (~4–8).  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as an integer matrix or list of integer vectors to avoid repeated string concatenation and lookups.
2. **Vectorize aggregation** using `data.table` or `dplyr` joins instead of per-row `lapply`.
3. **Process year-by-year** to reduce memory footprint (344k rows/year vs 6.46M at once).
4. **Use `data.table` rolling joins and grouping** for fast aggregation.
5. **Avoid repeated `do.call(rbind, ...)`**; compute in a single grouped operation.
6. Optionally, **parallelize by year** if multiple cores are available.

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure keys for fast joins
setkey(cell_data, id, year)

# Precompute neighbor pairs (id -> neighbor_id)
# rook_neighbors_unique: list of neighbors for each id in id_order
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand for all years (cartesian join with unique years)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Join neighbor values for all variables in one pass
for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
               by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Compute stats by (id, year)
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                               paste0(var_name, "_nbr_min"),
                               paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}
```

---

### **Why This Works**
- **No per-row loops**: Aggregation is done in compiled C via `data.table`.
- **Memory control**: Process year-by-year if needed:
  ```r
  for (yr in years) { ... subset cell_data[year == yr] ... }
  ```
- **Scales well**: Joins and aggregations on 6.5M rows are feasible in hours (or less) on 16 GB RAM.
- **Preserves model and estimand**: Only feature computation changes, not model training.

---

**Expected Performance**: From 86+ hours to a few hours or less, depending on disk speed and parallelization.