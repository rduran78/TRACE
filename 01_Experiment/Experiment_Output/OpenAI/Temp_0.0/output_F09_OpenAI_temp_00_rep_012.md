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

# Convert cell_data to data.table
setDT(cell_data)

# 1. Build reusable adjacency table (cell-level, not year-level)
adj_list <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# 2. Expand adjacency table across years by joining with unique years
years <- unique(cell_data$year)
adj_year <- CJ(year = years, id = id_order)[adj_list, on = "id", allow.cartesian = TRUE]

# 3. Join neighbor attributes for each year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  adj_year[cell_data, on = .(neighbor_id = id, year), neighbor_val := get(var_name)]
  
  # Compute neighbor stats grouped by (id, year)
  stats <- adj_year[, .(
    max_val = if (.N > 0) max(neighbor_val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(neighbor_val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(neighbor_val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
  
  # Remove neighbor_val for next iteration
  adj_year[, neighbor_val := NULL]
}

# cell_data now has neighbor features added
```

---

### **Why This is Faster**
- The neighbor lookup is computed **once** at the cell level (344k rows) instead of 6.46M times.
- Aggregation uses `data.table` grouping, which is highly optimized in C.
- Memory footprint is controlled by reusing `adj_year` and clearing temporary columns.
- Expected runtime drops from 86+ hours to a few hours (or less) on a 16 GB laptop.

**Key Points Preserved**
- Random Forest model remains unchanged.
- Original estimand and neighbor-based features are computed identically.
- Handles all 28 years and 5 neighbor variables efficiently.