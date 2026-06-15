 **Diagnosis**  
The bottleneck is the nested `lapply` structure in `build_neighbor_lookup` and repeated list processing in `compute_neighbor_stats`. For 6.46M rows, building a full neighbor index for every cell-year combination is extremely expensive in both time and memory. The current approach repeatedly constructs and looks up character keys (`paste(...)`) and processes lists, which is inefficient at this scale.

---

**Optimization Strategy**  
1. **Avoid repeated string concatenation and list indexing**: Precompute neighbor indices once at the cell level, then reuse them for all years.
2. **Vectorize neighbor feature computation**: Use matrix operations or `data.table` joins instead of per-row `lapply`.
3. **Process by year in chunks**: Compute neighbor stats year-by-year to keep memory manageable.
4. **Use `data.table` for fast joins and aggregation**: This avoids R loops and leverages optimized C code.
5. **Preserve the Random Forest model**: Only optimize feature construction.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor index mapping at cell level
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

setkey(neighbor_dt, neighbor_id)

# Function to compute neighbor stats for one variable, one year
compute_neighbor_stats_year <- function(dt_year, var_name) {
  # dt_year: data for one year
  setkey(dt_year, id)
  
  # Join neighbors to get neighbor values
  joined <- neighbor_dt[dt_year, on = .(neighbor_id = id), nomatch = 0]
  # joined: cell_id, neighbor_id, year vars from dt_year
  
  # Compute stats by cell_id
  stats <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = cell_id]
  
  # Merge back to dt_year
  dt_year[stats, on = .(id = cell_id),
          paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")) := .(i.max_val, i.min_val, i.mean_val)]
  
  dt_year
}

# Process year by year and all variables
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- vector("list", length(years))
for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  for (var_name in neighbor_source_vars) {
    dt_year <- compute_neighbor_stats_year(dt_year, var_name)
  }
  result_list[[i]] <- dt_year
}

# Combine back
cell_data <- rbindlist(result_list)
```

---

**Why this works**  
- Eliminates per-row loops: All neighbor computations are done via `data.table` joins and grouped aggregations.
- Avoids repeated string concatenation: Uses integer joins instead of key strings.
- Processes by year: Keeps memory usage within 16 GB.
- Reuses precomputed neighbor relationships efficiently.

**Expected speedup**: From 86+ hours to a few hours (or less) on a standard laptop, depending on disk I/O and CPU.