 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the code recomputes neighbor indices for every cell-year row. This results in billions of operations and excessive memory churn.

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use `data.table` for fast joins and aggregation** instead of nested `lapply`.  
4. **Compute neighbor stats in a single grouped operation** rather than row-wise loops.  

This avoids repeated string operations and leverages efficient joins and vectorized aggregation.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor lookup once per cell
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Melt cell_data for neighbor variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_data <- melt(cell_data, id.vars = c("id", "year"), 
                  measure.vars = neighbor_source_vars, 
                  variable.name = "var", value.name = "value")

# Join neighbor values
setkey(long_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
joined <- neighbor_dt[long_data, on = .(neighbor_id = id, year), allow.cartesian = TRUE]

# Compute neighbor stats by (id, year, var)
neighbor_stats <- joined[, .(
  max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
  min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
  mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
), by = .(id, year, var)]

# Reshape back to wide format
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var, 
                             value.var = c("max_val", "min_val", "mean_val"))

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- **No per-row string concatenation**: We use integer joins instead of `paste`.
- **Single pass aggregation**: `data.table` computes stats in compiled C loops.
- **Memory efficiency**: Avoids creating millions of small lists.
- **Scalability**: Handles 6.46M rows and 1.37M neighbor links efficiently on 16 GB RAM.

**Expected runtime**: From 86+ hours down to **minutes** (depending on disk I/O and CPU).  
**Preserves**: Original estimand, trained Random Forest model, and neighbor-based feature definitions.