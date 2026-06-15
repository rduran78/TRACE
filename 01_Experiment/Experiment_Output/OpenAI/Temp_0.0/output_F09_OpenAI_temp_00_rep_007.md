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

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor indices for each cell in id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# 1. Build reusable adjacency table (cell-level, static)
adj_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

setkey(adj_dt, id)

# Convert cell_data to data.table for speed
setDT(cell_data)

# 2. Compute neighbor stats for each year and variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in unique(cell_data$year)) {
  # Subset data for this year
  year_dt <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_dt, id)
  
  # Join adjacency with yearly data to get neighbor values
  joined <- adj_dt[year_dt, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined: id (focal), neighbor_id, year, vars...
  
  for (var_name in neighbor_source_vars) {
    stats <- joined[, .(
      max = max(get(var_name), na.rm = TRUE),
      min = min(get(var_name), na.rm = TRUE),
      mean = mean(get(var_name), na.rm = TRUE)
    ), by = .(id)]
    
    # Handle all-NA case
    stats[!is.finite(max), c("max", "min", "mean") := NA]
    
    # Merge back into cell_data
    setnames(stats, c("id", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
    cell_data[year == yr, (names(stats)[-1]) := stats[.SD, on = .(id), mget(names(stats)[-1])]]
  }
}

# At this point, cell_data has neighbor features added efficiently
# Proceed with Random Forest prediction using the existing trained model
```

---

### **Why This is Faster**
- Adjacency is computed **once** (344k rows, not 6.46M).
- For each year, we do a **single join** and **grouped aggregation** using `data.table` (highly optimized in C).
- Eliminates millions of list operations and repeated lookups.
- Memory footprint is controlled because we process one year at a time.

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU speed.