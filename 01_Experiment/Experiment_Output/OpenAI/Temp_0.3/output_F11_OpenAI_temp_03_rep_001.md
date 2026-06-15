 The evidence shows the main bottleneck is **neighbor feature computation**, not Random Forest inference.  
`build_neighbor_lookup()` and `compute_neighbor_stats()` perform millions of R-level list operations and repeated vector scans across **6.46M rows × 5 variables**, which is extremely slow in pure R. Random Forest `predict()` on 6.46M rows with 110 predictors is heavy but typically runs in minutes to a few hours on a laptop, not 86+ hours.  

### Diagnosis  
- `build_neighbor_lookup()` creates a large list of length 6.46M, with `lapply()` doing repeated string concatenation and name lookups.  
- `compute_neighbor_stats()` repeatedly subsets vectors and computes stats for each cell-year, again in R loops.  
- These operations dominate runtime due to R’s interpreter overhead and memory churn.  

### Optimization Strategy  
- Precompute neighbor indices **once** at the cell level, not per cell-year.  
- Use **vectorized joins** or **data.table** to compute neighbor stats in bulk instead of per-row loops.  
- Avoid repeated string concatenation and name-based lookups.  
- Keep Random Forest inference as-is.  

### Optimized R Code (data.table approach)  

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Add unique cell-year key
dt[, cell_year := .I]

# Precompute neighbor pairs at cell level
id_order_dt <- data.table(id = id_order, ref = seq_along(id_order))
neighbors_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(ref = i, neighbor_ref = rook_neighbors_unique[[i]])
}))

# Map to actual IDs
neighbors_dt <- merge(neighbors_dt, id_order_dt, by.x = "ref", by.y = "ref")
setnames(neighbors_dt, "id", "id")
neighbors_dt <- merge(neighbors_dt, id_order_dt, by.x = "neighbor_ref", by.y = "ref")
setnames(neighbors_dt, "id", "neighbor_id")

# Expand to cell-year pairs
years <- unique(dt$year)
neighbors_expanded <- neighbors_dt[, .(id, neighbor_id)][, .(year = years), by = .(id, neighbor_id)]
setkey(neighbors_expanded, neighbor_id, year)

# Join neighbor values and compute stats
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  tmp <- dt[, .(neighbor_id = id, year, val = get(var_name))]
  neighbors_expanded <- tmp[neighbors_expanded, on = .(neighbor_id, year)]
  neighbors_expanded[, (var_name) := val]
  neighbors_expanded[, val := NULL]
}

# Compute summary stats per id-year
stats <- neighbors_expanded[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Merge back into main data
setkey(stats, id, year)
setkey(dt, id, year)
dt <- stats[dt]
```

### Why This Works  
- Eliminates 6.46M `lapply()` calls and string operations.  
- Uses `data.table` joins and grouped aggregation (C-optimized).  
- Memory-efficient and vectorized: expected runtime drops from **86+ hours to under 1 hour** on a laptop.  

**Final:** The bottleneck is neighbor feature computation in R loops, not Random Forest inference. Optimize with `data.table` bulk joins and aggregations.