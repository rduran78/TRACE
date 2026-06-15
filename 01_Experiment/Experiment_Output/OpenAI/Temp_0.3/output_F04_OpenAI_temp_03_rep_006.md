 **Diagnosis**  
The bottleneck is the repeated R-level `lapply` loops over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs character concatenation, list indexing, and repeated lookups, which is extremely inefficient at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor indices for every cell-year row. This causes quadratic overhead.  

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell (not per cell-year)** and reuse them for all years.  
2. **Vectorize neighbor feature computation** by using `data.table` or `dplyr` joins instead of per-row `lapply`.  
3. **Avoid character key concatenation**; use integer indexing.  
4. **Compute stats in bulk**: reshape data to wide or use grouped joins to aggregate neighbor values efficiently.  
5. **Parallelize** using `data.table` or `future.apply` if needed.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor lookup as integer indices (once)
# id_order assumed to match unique(cell_data$id)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Expand neighbor relationships for all years
years <- sort(unique(cell_data$year))
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_list), function(i) {
  data.table(
    id = id_order[i],
    neighbor_id = id_order[neighbor_list[[i]]]
  )
}))
# Cartesian join with years
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setcolorder(neighbor_dt, c("id", "year", "neighbor_id"))

# Melt cell_data for fast joins
setkey(cell_data, id, year)

compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  tmp <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  setnames(tmp, var_name, "neighbor_val")
  
  # Aggregate stats by (id, year)
  agg <- tmp[, .(
    max = max(neighbor_val, na.rm = TRUE),
    min = min(neighbor_val, na.rm = TRUE),
    mean = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back
  setnames(agg, c("max", "min", "mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  cell_data[agg, on = .(id, year)]
}

# Apply for all variables
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(var_name)
}
```

**Why This Works**  
- Neighbor relationships are expanded once and reused for all variables.  
- Uses `data.table` joins and group aggregation instead of millions of `lapply` calls.  
- Eliminates string concatenation and repeated lookups.  
- Memory footprint is manageable: neighbor_dt ≈ 1.37M × 28 ≈ 38M rows (integer columns), feasible on 16 GB RAM with `data.table`.  

**Expected Speedup**  
From 86+ hours to a few hours (or less) on a standard laptop, depending on I/O and parallelization.