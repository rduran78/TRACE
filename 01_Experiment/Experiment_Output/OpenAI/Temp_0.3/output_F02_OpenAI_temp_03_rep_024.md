 **Diagnosis**  
The bottleneck arises from:  
- `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`.  
- Repeated string concatenation (`paste`) and list indexing.  
- Memory overhead from large lists and repeated `do.call(rbind, ...)`.  
- Pure R loops on a dataset of this scale are inefficient.  

**Optimization Strategy**  
- Avoid per-row string operations; precompute keys or use integer indexing.  
- Use `data.table` for fast joins and aggregations.  
- Vectorize neighbor statistics computation instead of looping.  
- Pre-expand neighbor relationships into a long table and compute stats via grouped operations.  
- Keep everything in memory-efficient structures (integers, no large lists).  

---

### **Optimized Approach**
1. Convert `cell_data` to `data.table`.
2. Expand neighbor relationships into a long table with `(cell_id, year, neighbor_id)`.
3. Join neighbor values for each variable and compute `max`, `min`, `mean` by `(cell_id, year)`.
4. Merge results back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbor_id)
# id_order: vector of cell IDs in order
# rook_neighbors_unique: list of integer neighbor indices
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i],
             neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years, cell_id = cell_id, unique = TRUE), on = "cell_id"]
setnames(neighbor_dt, "year", "year")  # ensure column name consistency

# Add neighbor_id-year pairs
neighbor_dt[, neighbor_year := year]

# Join neighbor values for each variable and compute stats
compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  tmp <- merge(neighbor_dt,
               cell_data[, .(neighbor_id = id, year, val = get(var_name))],
               by.x = c("neighbor_id", "neighbor_year"),
               by.y = c("neighbor_id", "year"),
               all.x = TRUE)
  
  # Compute stats by (cell_id, year)
  tmp_stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]
  
  # Rename columns
  setnames(tmp_stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  tmp_stats
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features and merge
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_features(var_name)
  cell_data <- merge(cell_data, stats_dt, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
}
```

---

### **Why This Works**
- Eliminates per-row loops: uses `data.table` joins and group operations.
- Avoids string concatenation for keys.
- Processes all years and neighbors in bulk.
- Memory footprint is manageable because:
  - `neighbor_dt` size ≈ 1.37M * 28 ≈ 38M rows (integers, efficient in `data.table`).
- Expected runtime: **minutes to a few hours** instead of 86+ hours.

**Preserves:**  
- Original numerical estimand.  
- Trained Random Forest model (unchanged).  

**Tip:** If memory is still tight, process one variable at a time (as shown) or split by year chunks.