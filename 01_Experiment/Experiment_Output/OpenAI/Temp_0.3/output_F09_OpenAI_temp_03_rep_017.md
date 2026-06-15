 **Diagnosis**  
The current pipeline is slow because for each of ~6.46M rows, `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly compute neighbor indices and statistics. This involves heavy repeated string concatenation (`paste`), repeated lookups in large named vectors, and repeated list traversals across 28 years. The complexity is roughly `O(N * neighbors)` for each variable, multiplied by 5 variables, causing massive overhead (86+ hours).  

**Optimization Strategy**  
- Build a **reusable adjacency table** once, mapping each cell-year row to its neighbor row indices for all years.
- Avoid repeated string operations; instead, use integer joins keyed by `(id, year)`.
- Compute neighbor stats in a **vectorized** way by reshaping data into a matrix by year and applying fast row-wise aggregations.
- Use `data.table` for efficient joins and grouping.
- Preserve the trained Random Forest model and original estimand.

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute adjacency table (cell-to-neighbor relationships)
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
adj_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand adjacency across years (Cartesian join on year)
years <- sort(unique(cell_data$year))
adj_dt <- adj_dt[, .(id = rep(id, each = length(years)),
                     neighbor_id = rep(neighbor_id, each = length(years)),
                     year = rep(years, times = .N))]
# Result: ~1.37M * 28 rows (~38M rows) but manageable with data.table

# Key for fast joins
setkey(cell_data, id, year)
setkey(adj_dt, neighbor_id, year)

# Join neighbor attributes onto adjacency table
adj_dt <- cell_data[adj_dt, on = .(id = neighbor_id, year), nomatch = 0]

# Now adj_dt has columns: id (focal), year, neighbor_id, var values

# Compute neighbor stats for each variable efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- adj_dt[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added

# Predict with existing Random Forest model (already trained)
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This is Faster**
- **One-time adjacency expansion**: No repeated string concatenation or list traversal.
- **Vectorized aggregation**: `data.table` computes stats in C-optimized loops.
- **Memory efficiency**: 38M rows is large but feasible on 16 GB RAM with `data.table`.
- **No retraining**: Original Random Forest model is preserved.

**Expected runtime improvement**: From 86+ hours to a few hours (or less) depending on disk I/O and CPU.