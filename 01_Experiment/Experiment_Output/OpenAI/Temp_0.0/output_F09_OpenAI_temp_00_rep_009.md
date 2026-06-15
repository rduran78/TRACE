 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly perform expensive lookups and list operations. The neighbor relationships are static across years, but the code recomputes neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

**Optimization Strategy**  
- Precompute a reusable adjacency table (cell-to-neighbor mapping) **once** at the cell level (344,208 cells).
- For each year, join cell attributes to this adjacency table and compute neighbor statistics using **vectorized operations** (e.g., `data.table` joins or `dplyr` group operations).
- Avoid per-row `lapply` loops; instead, use grouped aggregation.
- Keep the Random Forest model unchanged and ensure the estimand remains the same.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute adjacency table once
# rook_neighbors_unique: list of neighbor indices for each cell in id_order
adj_list <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# Ensure keys for fast joins
setkey(adj_list, neighbor_id)
setkey(cell_data, id)

# Function to compute neighbor stats for all years and one variable
compute_neighbor_stats_fast <- function(cell_data, adj_list, var_name) {
  # Select relevant columns
  dt <- cell_data[, .(id, year, value = get(var_name))]
  
  # Join adjacency table to get neighbor values
  joined <- adj_list[cell_data, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined now has: cell_id, neighbor_id, year, value
  
  # Compute stats by cell_id and year
  stats <- joined[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Handle cases where all neighbors are NA
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back to main data
  setnames(stats, c("cell_id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats, on = .(id = cell_id, year)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_fast(cell_data, adj_list, var_name)
}

# cell_data now has neighbor features added
```

---

### **Why This Is Faster**
- Adjacency is computed **once** (344k cells, ~1.37M neighbor pairs).
- For each variable, we do a **single join and grouped aggregation** instead of millions of list lookups.
- `data.table` operations are highly optimized for large datasets.
- Memory footprint is reduced because we avoid storing massive lists of indices for every row-year.

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU speed.