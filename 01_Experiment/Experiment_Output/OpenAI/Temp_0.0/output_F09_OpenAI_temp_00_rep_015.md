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

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute adjacency table once
# rook_neighbors_unique: list of integer vectors (neighbors for each cell)
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(lapply(seq_along(adj_list), function(i) {
  if (length(adj_list[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
}))

# Convert to data.table for fast joins
setkey(adj_dt, neighbor_id)
setkey(cell_data, id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Join neighbor attributes for the same year
  joined <- adj_dt[cell_data, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined now has: id (from adj_dt), neighbor_id, year, var_name
  setnames(joined, var_name, "neighbor_val")
  
  # Aggregate by id and year
  stats <- joined[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle all-NA cases (replace Inf/-Inf with NA)
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_fast(var_name)
}

# cell_data now has neighbor features added
# Proceed with Random Forest prediction using the existing trained model
```

---

### **Why This Is Faster**
- Adjacency is computed **once** (344k rows, not 6.46M).
- Joins and aggregations are vectorized and use `data.table` for efficiency.
- No nested loops over millions of rows.
- Memory footprint is controlled by processing one variable at a time.

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU speed.