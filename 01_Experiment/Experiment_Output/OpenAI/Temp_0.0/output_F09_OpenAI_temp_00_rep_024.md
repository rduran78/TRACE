 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly perform expensive lookups and list operations. The neighbor relationships are static across years, but the code recomputes neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

**Optimization Strategy**  
- Precompute a reusable adjacency table (cell-to-neighbor mapping) **once** at the cell level (344,208 cells).
- For each year, join cell attributes to this adjacency table and compute neighbor statistics using fast vectorized operations (`data.table` or `dplyr`), rather than looping through millions of rows.
- Avoid repeated list traversals; instead, use a long-format join and grouped aggregation.
- Keep the Random Forest model unchanged and ensure the estimand remains the same.

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor indices for each cell in id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert cell_data to data.table
setDT(cell_data)

# 1. Build reusable adjacency table (cell-level, static)
adj_list <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# 2. Expand adjacency table across years by joining with unique years
years <- unique(cell_data$year)
adj_dt <- adj_list[CJ(year = years, id = id, unique = TRUE), on = "id"]
setnames(adj_dt, "id", "cell_id")

# 3. Join neighbor attributes for each year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  adj_dt <- tmp[adj_dt, on = .(neighbor_id, year)]
  
  # Compute stats by cell_id and year
  stats <- adj_dt[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Handle all-NA cases
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back into cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(cell_id = id, year)]
}

# cell_data now has neighbor features added efficiently
```

---

### **Why This is Faster**
- Adjacency is computed **once** (344k rows, not 6.46M).
- Year expansion and joins are vectorized using `data.table`.
- Aggregations are done in grouped operations rather than millions of list traversals.
- Memory footprint is controlled by processing one variable at a time.

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and disk speed.

**Preserves:**  
- Original Random Forest model (unchanged).
- Original estimand (neighbor max, min, mean for each variable).