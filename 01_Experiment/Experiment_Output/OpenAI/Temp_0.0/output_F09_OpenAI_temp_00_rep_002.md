 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` repeatedly constructs neighbor indices by string concatenation and hash lookups. This is done once, but then `compute_neighbor_stats` iterates over all rows and performs repeated list indexing and aggregation. The complexity is roughly *O(N × avg_neighbors)* per variable, repeated for 5 variables, leading to billions of operations in R lists. Memory overhead and R’s interpreted loops exacerbate the problem.

**Optimization Strategy**  
- Precompute a **static adjacency table** (cell_id → neighbor_id) once, independent of year.
- Expand this adjacency to a **cell-year → neighbor-year** mapping by joining on year, avoiding per-row string concatenation.
- Use **data.table** for fast joins and grouped aggregations instead of R loops.
- Compute neighbor max, min, mean in a single grouped operation per variable per year.
- Reuse the trained Random Forest model without retraining.

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute adjacency table (cell_id, neighbor_id)
# rook_neighbors_unique: list of integer vectors, same order as id_order
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(lapply(seq_along(adj_list), function(i) {
  if (length(adj_list[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
}))

# Expand to cell-year adjacency by joining on year
years <- unique(cell_data$year)
adj_expanded <- adj_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(adj_expanded, "year", "year")  # now adj_expanded has id, neighbor_id, year

# Join neighbor attributes for each year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor values
  tmp <- merge(adj_expanded,
               cell_data[, .(neighbor_id = id, year, val = get(var))],
               by = c("neighbor_id", "year"),
               all.x = TRUE)
  
  # Aggregate neighbor stats by (id, year)
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor features added efficiently
```

---

**Why This Is Faster**  
- Eliminates per-row loops and string concatenation.
- Uses `data.table` joins and aggregations, which are highly optimized in C.
- Adjacency is computed once and reused for all variables.
- Complexity reduced to roughly *O(N + E)* per variable, where E ≈ 1.37M edges × 28 years ≈ 38M rows, which is manageable on a 16 GB machine.

**Expected Runtime**  
From 86+ hours down to a few hours (or less) depending on disk I/O and CPU speed, because the bottleneck is now vectorized joins and aggregations rather than R loops.

**Preserves**  
- Original estimand (neighbor max, min, mean).
- Trained Random Forest model (no retraining).
- All original predictor variables plus new neighbor features.