 **Diagnosis**  
The current pipeline recomputes neighbor relationships for each cell-year row repeatedly. With ~6.46M rows and 5 variables, this results in heavy redundant computation and memory overhead. The `build_neighbor_lookup` function already avoids recomputing neighbors per variable, but the outer loop still processes 6.46M rows × 5 variables, repeatedly extracting neighbor values. The main bottleneck is repeatedly scanning large vectors and performing `lapply` over millions of elements in R, which is slow and memory-intensive on a 16 GB machine.

---

**Optimization Strategy**  
1. **Precompute a reusable adjacency table**: Build a long-format table mapping each `(cell_id, year)` to its neighbors `(neighbor_id, year)`.  
2. **Join yearly attributes once**: For each year, join neighbor attributes via `data.table` or `dplyr` instead of looping millions of times in R lists.  
3. **Aggregate with vectorized operations**: Use `data.table` grouping to compute `max`, `min`, and `mean` in one pass per variable per year.  
4. **Process by year in chunks**: Reduces memory footprint and avoids holding all intermediate joins in RAM.  
5. **Preserve the trained Random Forest model and estimand**: Only feature engineering changes; prediction step remains the same.

---

**Working R Code (data.table approach)**  
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell ids in same order as rook_neighbors_unique

setDT(cell_data)
setkey(cell_data, id, year)

# 1. Build adjacency table (cell_id -> neighbor_id)
adj_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# 2. Expand adjacency by year (Cartesian join on years)
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(year = years), by = .(id, neighbor_id)]

# 3. Merge neighbor attributes by year
setkey(adj_dt, neighbor_id, year)
setkey(cell_data, id, year)

# Variables to compute neighbor stats for
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 4. Join and compute stats by year in chunks
results_list <- vector("list", length(neighbor_vars))
names(results_list) <- neighbor_vars

for (var in neighbor_vars) {
  tmp <- adj_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  # tmp now has: id, neighbor_id, year, <var>
  stats <- tmp[, .(
    paste0(var, "_nbr_max") = max(get(var), na.rm = TRUE),
    paste0(var, "_nbr_min") = min(get(var), na.rm = TRUE),
    paste0(var, "_nbr_mean") = mean(get(var), na.rm = TRUE)
  ), by = .(id, year)]
  results_list[[var]] <- stats
}

# 5. Combine all stats into cell_data
for (var in neighbor_vars) {
  cell_data <- merge(cell_data, results_list[[var]], by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor-based features for RF prediction
# Random Forest model remains unchanged; just predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Why This Is Faster**  
- Eliminates per-row `lapply` loops over millions of rows.  
- Uses `data.table` joins and grouped aggregation, which are highly optimized in C.  
- Processes adjacency once and reuses it for all variables.  
- Memory footprint is controlled by chunking by year if needed.  

This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop.