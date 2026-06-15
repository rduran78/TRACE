 **Diagnosis**  
The bottleneck is indeed the neighbor feature construction, not Random Forest inference. The current implementation repeatedly uses `lapply` over ~6.46M rows, building neighbor indices and computing statistics in pure R loops. This is extremely inefficient because:

- `build_neighbor_lookup` constructs a large nested list of integer vectors via R loops.
- `compute_neighbor_stats` iterates again through all rows for each variable (5 times), performing repeated lookups and aggregations.
- Memory footprint and function call overhead dominate performance.

Given 6.46M rows and 5 variables, this approach does tens of millions of small R operations.  

---

**Optimization Strategy**  
1. **Vectorize neighbor relationships**: Flatten neighbor pairs into a long data frame (cell-year row → neighbor row) once.
2. **Use `data.table` for aggregation**: Compute max/min/mean per focal row across neighbors efficiently in C-optimized code.
3. **Reuse join structure**: Instead of recalculating per variable, join values in one pass and compute summaries grouped by focal row.
4. **Parallelize aggregation**: Use `data.table` multithreading (built-in) or `future.apply` if necessary.
5. **Avoid storing huge lists**: Work with integer vectors and keyed joins instead.

This converts the problem into a large but efficient grouped aggregation task.

---

**Optimized R Code using `data.table`**  
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor indices for each cell (by id_order)

# Convert cell_data to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Step 1: Build long neighbor table (cell-year to neighbor cell-year)
id_order <- unique(cell_data$id)
id_to_idx <- setNames(seq_along(id_order), id_order)

# Precompute all neighbor pairs (cell -> neighbors)
pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand pairs across years (cartesian join)
years <- sort(unique(cell_data$year))
pairs_year <- pairs[, .(id, neighbor_id), keyby = .(id)][
  , .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Map to row indices
setkey(cell_data, id, year)
pairs_year[, focal_idx := cell_data[.(id, year), which = TRUE]]
pairs_year[, neigh_idx := cell_data[.(neighbor_id, year), which = TRUE]]

# Drop missing
pairs_year <- pairs_year[!is.na(focal_idx) & !is.na(neigh_idx)]

# Step 2: Compute neighbor stats for all vars in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  pairs_year[, value := vals[neigh_idx]]
  
  agg <- pairs_year[!is.na(value),
    .(max_val = max(value), min_val = min(value), mean_val = mean(value)),
    by = focal_idx]
  
  # Add back to main table
  cell_data[agg$focal_idx,
            paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")) :=
              .(agg$max_val, agg$min_val, agg$mean_val)]
}

# cell_data now has neighbor features for all variables
```

---

**Why this is faster**  
- `data.table` aggregation operates in C with multi-threading.
- We avoid building huge nested lists.
- Only one neighbor mapping pass; then reuse for all variables.
- Memory-efficient: works with integer indices and vectorized joins.

---

**Expected result**  
- Runtime reduced from 86+ hours to a few hours (or less) on a 16 GB machine.
- Preserves the original numerical estimand and Random Forest model.
- Scalable to millions of rows and large neighbor sets.