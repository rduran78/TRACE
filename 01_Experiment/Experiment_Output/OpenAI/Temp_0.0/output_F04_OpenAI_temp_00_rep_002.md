 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use `data.table` for fast joins and aggregation** instead of nested `lapply`.  
4. **Compute neighbor stats in a single grouped operation** rather than row-wise loops.  

This avoids 6.46M × 5 loops and reduces complexity from O(N × neighbors) string operations to O(N) numeric joins.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure keys
setkey(cell_data, id, year)

# Precompute neighbor list as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- lapply(rook_neighbors_unique, function(nbs) id_order[nbs])

# Expand neighbor relationships across years
years <- sort(unique(cell_data$year))
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(neighbor_list[[i]]) == 0) return(NULL)
  data.table(
    id = rep(id_order[i], length(neighbor_list[[i]]) * length(years)),
    year = rep(years, each = length(neighbor_list[[i]])),
    neighbor_id = rep(neighbor_list[[i]], times = length(years))
  )
}))

setkey(neighbor_dt, neighbor_id, year)

# Join neighbor values for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(v))]
  setkey(tmp, neighbor_id, year)
  
  neighbor_vals <- neighbor_dt[tmp, on = .(neighbor_id, year)]
  
  stats <- neighbor_vals[, .(
    max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  setnames(stats, c("id", "year", paste0(v, "_nb_max"), paste0(v, "_nb_min"), paste0(v, "_nb_mean")))
  
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added
```

---

**Why This Works**  
- Eliminates per-row string concatenation and hash lookups.
- Uses `data.table` joins and grouped aggregation for speed and memory efficiency.
- Precomputes neighbor relationships once and reuses them for all variables.
- Preserves the original estimand and Random Forest model.

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to a few hours or less, depending on disk I/O and CPU speed. Further speedups possible via parallel `data.table` or `fst` for fast serialization.