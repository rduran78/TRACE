 **Diagnosis:**  
The bottleneck is the repeated neighbor lookups for each cell-year row (≈6.46M rows) using dynamic computation inside `compute_neighbor_stats`. This causes enormous overhead because the neighbor structure is static across years, yet it's recomputed for every row. Memory thrashing and R's lapply loops further slow execution.

---

**Optimization Strategy:**  
- **Precompute static neighbor mapping** at the *cell level* (not cell-year).  
- For each year, compute neighbor statistics **vectorized** by joining data frames instead of looping through 6.46M rows.  
- Use **data.table** or **dplyr** for efficient grouping and joining.  
- Avoid recomputing for each variable separately—compute all neighbor stats in a single pass if possible.  

---

**Optimized R Code:**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# Precompute static neighbor list for each cell id
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Reshape data for fast joins
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join by year for neighbor values
# Expand neighbor relationships across years
years <- unique(cell_data$year)
neighbor_year_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_year_dt, "year", "year")
neighbor_year_dt[, dummy := NULL]

# Merge cell_data for neighbor values
cell_data_long <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                       variable.name = "var", value.name = "val")

neighbor_vals <- merge(neighbor_year_dt, cell_data_long,
                       by.x = c("neighbor_id", "year"), by.y = c("id", "year"),
                       allow.cartesian = TRUE)

# Compute stats per (id, year, var)
neighbor_stats <- neighbor_vals[, .(
  neigh_max = max(val, na.rm = TRUE),
  neigh_min = min(val, na.rm = TRUE),
  neigh_mean = mean(val, na.rm = TRUE)
), by = .(id, year, var)]

# Reshape back to wide
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var,
                              value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why it works:**  
- Neighbor relationships (≈1.37M) are expanded once per year (28 years → ~38M rows), which is large but manageable with `data.table` instead of nested loops across 6.46M rows.  
- All variables are processed in a single grouped aggregation, avoiding repeated joins.  
- Preserves original estimand and integrates seamlessly before the Random Forest prediction step.  

**Expected runtime:** Hours reduced to **minutes** on a 16 GB laptop due to vectorization and avoidance of repeated lapply loops.