 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly perform expensive lookups and list operations. The neighbor relationships are static across years, but the code recomputes neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

**Optimization Strategy**  
- Precompute a reusable adjacency table (cell-to-neighbor mapping) **once** at the cell level (344,208 cells).
- For each year, join cell attributes to this adjacency table and compute neighbor statistics using fast vectorized operations (`data.table` or `dplyr`), rather than looping through millions of rows.
- Avoid repeated `lapply` over 6.46M rows; instead, use grouped aggregation.
- Keep the Random Forest model unchanged and ensure the estimand remains the same.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute adjacency table once (cell-level)
# rook_neighbors_unique: list of neighbor indices for each cell in id_order
adj_list <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# Ensure keys for fast joins
setkey(cell_data, id, year)
setkey(adj_list, cell_id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Join cell_data (as neighbors) to adjacency table
  neighbor_vals <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)
  
  # Join adjacency with neighbor values
  adj_year <- adj_list[cell_data[, .(cell_id = id, year)], allow.cartesian = TRUE]
  adj_year <- neighbor_vals[adj_year, on = .(neighbor_id, year)]
  
  # Compute stats by cell_id-year
  stats <- adj_year[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]
  
  # Merge back to main data
  setkey(stats, cell_id, year)
  cell_data[stats, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats, paste0(var_name, "_nbr_mean") := i.mean_val]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}

# cell_data now has neighbor features added
# Proceed with Random Forest prediction using the existing trained model
```

---

**Why This Is Faster**  
- Adjacency is computed once (344k cells, not 6.46M rows).
- Yearly joins and aggregations are vectorized using `data.table`, avoiding millions of R-level loops.
- Memory footprint is reduced by avoiding large nested lists.

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to a few hours or less, depending on disk I/O and CPU speed.