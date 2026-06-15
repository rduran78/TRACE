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

# 1. Build reusable adjacency table (cell-level, no years)
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]))
  data.table(id = from, neighbor_id = to)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)
setkey(adj_table, neighbor_id)

# Convert cell_data to data.table for speed
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# 2. Compute neighbor stats for each year and variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join adjacency with yearly data for neighbors
  tmp <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(tmp, neighbor_id, year)
  
  # Join to adjacency table
  joined <- adj_table[tmp, on = .(neighbor_id), nomatch = 0]
  # joined now has: id (focal), neighbor_id, year, val
  
  # Aggregate neighbor stats by focal id and year
  stats <- joined[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle cases where all neighbors are NA
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back to main data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_dt <- stats[cell_dt, on = .(id, year)]
}

# cell_dt now has neighbor features added efficiently
# Proceed with Random Forest prediction using the existing trained model
```

---

### **Why This Is Faster**
- Adjacency is computed **once** (≈1.37M rows vs. 6.46M × 5 loops).
- Yearly joins and aggregations are vectorized using `data.table` instead of R loops.
- Memory footprint is reduced by avoiding large nested lists.
- Expected runtime drops from 86+ hours to a few hours (or less) on a 16 GB laptop.

**Result:** Same estimand, same model, but dramatically faster feature computation.