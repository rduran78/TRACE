 **Diagnosis:**  
The computational bottleneck is in `build_neighbor_lookup` and `compute_neighbor_stats`. Both use `lapply` over ~6.46M rows, repeatedly doing vector lookups and string concatenation (`paste`). These operations are extremely costly at this scale. Random Forest inference is negligible compared to this feature construction.

---

**Optimization Strategy:**  
1. **Avoid repeated `paste()` and name-based lookups**: Switch to integer indices for direct access.  
2. **Precompute neighbor indices for all cell-years in a vectorized/data.table manner** rather than per-row `lapply`.  
3. **Use matrix/data.table joins instead of millions of small R closures**: Compute neighbor stats by grouping rather than iterating.  
4. **Leverage `data.table` for speed and memory efficiency**: It handles 6.5M rows comfortably on 16 GB RAM.  

---

**Optimized R Code (data.table approach):**

```r
library(data.table)

# Assume cell_data is a data.table: columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor relationships expanded by year
# rook_neighbors_unique: list of integer vectors (neighbors) indexed by cell id order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    src_id = id_order[i],
    nb_id  = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years (Cartesian join)
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = src_id, nb_id, year = years), by = .(src_id, nb_id)]

# Merge neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, nb_id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(nb_id = id, year)]

# Now compute neighbor stats for each src_id-year
compute_neighbor_stats_dt <- function(var) {
  neighbor_dt[, .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = .(id, year)]
}

# Compute and merge all neighbor features
for (var in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  stats_dt <- compute_neighbor_stats_dt(var)
  setnames(stats_dt, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var, "_nb_max"),
             paste0(var, "_nb_min"),
             paste0(var, "_nb_mean")))
  cell_data <- stats_dt[cell_data, on = .(id, year)]
}
```

---

**Why this is faster:**  
- Eliminates millions of `lapply` calls and string operations.
- Uses efficient keyed joins and grouped aggregations in `data.table`.
- Handles all variables in bulk without recomputing neighbor lookups.
- Scales to millions of rows within hours rather than days.

**Expected result:**  
Runtime drops from 86+ hours to a few hours on a standard laptop, with the same numerical estimand and no retraining of the Random Forest model.