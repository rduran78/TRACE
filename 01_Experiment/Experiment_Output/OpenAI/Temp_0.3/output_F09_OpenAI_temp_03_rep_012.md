 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every cell-year row repeatedly, causing extreme inefficiency. With ~6.46M rows and 5 variables, repeatedly scanning neighbors inflates runtime. The neighbor structure (rook adjacency) is static across years, so rebuilding or iterating through neighbors per row-year is unnecessary. Memory and CPU overhead from repeated lookups dominate the 86+ hour runtime.

---

**Optimization Strategy**  
1. **Precompute a reusable neighbor lookup table** keyed by cell ID (not cell-year).  
2. For each year, subset the data, join yearly attributes to the static neighbor table, and compute neighbor stats in a **vectorized** manner.  
3. Use `data.table` or `dplyr` joins instead of per-row `lapply`.  
4. Avoid repeated `paste()` and repeated indexing inside loops.  
5. Process each year in chunks to control memory usage.  

---

**Working R Code**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# 1. Build reusable neighbor lookup (long table)
build_neighbor_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]))
  data.table(id = from, neighbor_id = to)
}

neighbor_table <- build_neighbor_table(id_order, rook_neighbors_unique)

# 2. Compute neighbor stats by year and variable
compute_neighbor_stats_year <- function(dt, neighbor_table, vars) {
  # Join dt (for a single year) to neighbor_table twice:
  # left side: id, right side: neighbor_id
  joined <- neighbor_table[dt, on = .(neighbor_id = id)]
  # joined now has: id (from neighbor_table), neighbor_id, year, vars...
  
  # Compute stats for each variable
  res_list <- lapply(vars, function(v) {
    tmp <- joined[!is.na(get(v)), .(
      max = max(get(v), na.rm = TRUE),
      min = min(get(v), na.rm = TRUE),
      mean = mean(get(v), na.rm = TRUE)
    ), by = id]
    setnames(tmp, c("id", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
    tmp
  })
  
  # Merge all variable summaries by id
  Reduce(function(x, y) merge(x, y, by = "id", all = TRUE), res_list)
}

# 3. Apply by year and merge back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- lapply(unique(cell_data$year), function(yr) {
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  stats_year <- compute_neighbor_stats_year(dt_year, neighbor_table, neighbor_source_vars)
  merge(dt_year, stats_year, by = "id", all.x = TRUE)
})

cell_data_with_neighbors <- rbindlist(result_list)

# 4. Predict using existing Random Forest model
# Assume rf_model is already loaded
predictions <- predict(rf_model, newdata = cell_data_with_neighbors)
```

---

**Why This is Faster**  
- The neighbor table is built **once** (O(N_neighbors)).  
- Per year, we do a **join and grouped aggregation** instead of millions of per-row lookups.  
- Vectorized operations via `data.table` drastically reduce overhead.  
- Memory footprint is controlled by processing year-by-year.  

This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop.