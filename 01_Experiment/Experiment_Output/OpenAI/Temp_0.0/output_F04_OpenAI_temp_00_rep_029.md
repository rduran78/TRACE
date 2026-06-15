 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, hash lookups, and list indexing in R, which is extremely slow at this scale. The neighbor structure is static across years, but the current code rebuilds neighbor indices for every row-year combination. This results in massive overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use `data.table` for fast joins and aggregation** instead of `lapply`.  
4. **Compute neighbor stats in a single grouped operation** rather than looping over rows.  

This avoids 6.46M iterations and leverages efficient joins and grouping.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor lookup as a list of integer indices (cell-level)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Expand neighbor relationships to all years
years <- sort(unique(cell_data$year))
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[neighbor_lookup[[i]]]
  )
}))
# Cartesian join with years
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setcolorder(neighbor_dt, c("id", "year", "neighbor_id"))

# Melt cell_data for fast joins
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  # Join neighbor values
  tmp <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
               by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Compute stats by (id, year)
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("id", "year", paste0(var_name, "_nb_max"), paste0(var_name, "_nb_min"), paste0(var_name, "_nb_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor features added
```

---

**Why This Works**  
- Eliminates per-row string operations and `lapply` loops.
- Uses `data.table` joins and group aggregation, which are highly optimized in C.
- Precomputes neighbor relationships once and reuses them for all years.
- Memory footprint remains manageable because neighbor_dt ≈ 1.37M * 28 ≈ 38M rows, which fits in 16 GB RAM with efficient storage.

**Expected Speedup**: From 86+ hours to under 1 hour on a standard laptop.