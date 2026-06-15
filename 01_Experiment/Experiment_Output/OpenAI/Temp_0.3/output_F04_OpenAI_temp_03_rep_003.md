 **Diagnosis**  
The bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. For each row, the code constructs neighbor keys and performs string-based lookups, causing massive overhead. The neighbor structure is static across years, but the current implementation recomputes neighbor indices per row and per variable, leading to redundant work and poor memory locality.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once for all years**: Expand the static neighbor graph across time without repeated string concatenation.
2. **Avoid per-row `lapply`**: Use vectorized operations or `data.table` joins for speed.
3. **Compute all neighbor stats in one pass**: Instead of looping over variables, compute max/min/mean for all required variables in a single grouped operation.
4. **Use `data.table` for efficient joins and aggregations**.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure keys for fast join
setkey(cell_data, id, year)

# Precompute neighbor pairs across all years
# rook_neighbors_unique: list of neighbors per cell id in id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to replicate for panel
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Join neighbor values
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# Compute stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_stats <- neighbor_dt[, lapply(.SD, function(x) {
  .(max = max(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    mean = mean(x, na.rm = TRUE))
}), by = .(id, year), .SDcols = neighbor_source_vars]

# Reshape wide: each var gets _max/_min/_mean
agg_stats <- melt(agg_stats, id.vars = c("id", "year"), variable.name = "var", value.name = "val", variable.factor = FALSE)
agg_stats[, c("var", "stat") := tstrsplit(var, "\\.", fixed = TRUE)]
agg_stats <- dcast(agg_stats, id + year ~ var + stat, value.var = "val")

# Merge back to cell_data
cell_data <- merge(cell_data, agg_stats, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- Eliminates per-row string operations and multiple loops.
- Uses `data.table` joins and aggregations for high performance.
- Computes all neighbor features in one grouped aggregation.
- Preserves the original estimand and does not retrain the Random Forest model.

**Expected Speedup**: From 86+ hours to under 1 hour on a 16 GB RAM laptop (depending on disk I/O).