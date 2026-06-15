 **Diagnosis**  
The bottleneck is the repeated use of `lapply` over ~6.46M rows and dynamic lookups in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, list indexing, and repeated memory allocations. This is highly inefficient for large panel data. The process is CPU-bound and memory-intensive because it builds neighbor indices row-by-row and computes stats repeatedly for each variable.

**Optimization Strategy**  
1. **Precompute neighbor indices by (cell_id, year)** using integer mapping instead of string keys.
2. **Vectorize computations**: Avoid per-row `lapply` by using `data.table` or `dplyr` joins and grouped aggregations.
3. **Use long format and keyed joins**: Expand neighbor relationships across years once, then compute stats in bulk.
4. **Avoid repeated loops for each variable**: Compute all neighbor stats in one pass.
5. **Memory efficiency**: Use integer vectors and `data.table` for fast joins and aggregation.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
cell_dt <- as.data.table(cell_data)

# Precompute neighbor relationships across years
# id_order: vector of cell IDs in reference order
# rook_neighbors_unique: list of neighbor indices per cell
neighbor_pairs <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand across years
years <- unique(cell_dt$year)
neighbor_dt <- neighbor_pairs[, .(id, neighbor_id), by = .(year = years)]

# Join to get neighbor values
setkey(cell_dt, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Merge neighbor values for all source vars
merged_dt <- neighbor_dt[cell_dt, on = .(neighbor_id = id, year), nomatch = 0]

# Compute stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats_dt <- merged_dt[, lapply(.SD, function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(NA_real_, NA_real_, NA_real_))
  c(max(x), min(x), mean(x))
}), by = .(id, year), .SDcols = neighbor_source_vars]

# Reshape stats into wide format with suffixes
stat_names <- c("max", "min", "mean")
stats_long <- melt(stats_dt, id.vars = c("id", "year"))
stats_long[, c("var", "stat") := tstrsplit(variable, "_", fixed = TRUE)]
stats_wide <- dcast(stats_long, id + year ~ var + stat, value.var = "value")

# Merge back to original data
cell_dt <- merge(cell_dt, stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works**  
- Eliminates per-row loops: All neighbor relationships are expanded once.
- Uses `data.table` joins and aggregation for speed and memory efficiency.
- Computes all neighbor stats in one grouped operation instead of 5 separate loops.
- Avoids string concatenation and repeated indexing.

**Expected Performance**  
This approach reduces complexity from O(N × neighbors × vars) with heavy R overhead to efficient keyed joins and grouped aggregation. On 6.46M rows, this should complete in **minutes to a few hours** on a 16 GB laptop instead of 86+ hours.

**Preserves**  
- Original numerical estimand.
- Trained Random Forest model (no retraining).
- All required neighbor-based features.