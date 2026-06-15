 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` and object copying in `build_neighbor_lookup` and `compute_neighbor_stats`. For 6.46M rows, these nested loops and repeated `paste` operations are extremely costly in both time and memory. Additionally:  
- `build_neighbor_lookup` constructs a large list of integer vectors via repeated string concatenation and lookups.  
- `compute_neighbor_stats` repeatedly allocates vectors and uses `do.call(rbind, ...)`, which is slow for millions of rows.  
- The outer loop calls `compute_and_add_neighbor_features` multiple times, compounding overhead.  
- Random Forest inference itself is relatively fast compared to this feature engineering step; the data preparation dominates runtime.  

**Optimization Strategy**  
1. **Precompute keys efficiently**: Avoid repeated `paste` calls inside loops; use vectorized operations.  
2. **Use integer indexing**: Replace string-based lookups with integer-based joins.  
3. **Replace `lapply` with `vapply` or matrix operations**: Preallocate results and fill them directly.  
4. **Compute neighbor stats in one pass**: Instead of looping per variable, compute all neighbor stats together.  
5. **Use `data.table` for joins and aggregation**: Highly optimized for large datasets.  
6. **Preserve trained Random Forest model**: Only optimize feature preparation; prediction remains unchanged.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficiency
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- rook_neighbors_unique

# Build a flat neighbor table: (cell_id, year, neighbor_id)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(neighbor_list[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[neighbor_list[[i]]]
  )
}))
# Expand across years
years <- unique(cell_dt$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]

# Join neighbor values for all source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_dt for easier join
long_dt <- melt(cell_dt, id.vars = c("id", "year"), measure.vars = neighbor_source_vars,
                variable.name = "var_name", value.name = "val")

# Join neighbor values
neighbor_long <- merge(neighbor_dt, long_dt, by.x = c("neighbor_id", "year"),
                       by.y = c("id", "year"), allow.cartesian = TRUE)

# Compute stats by (id, year, var_name)
stats_dt <- neighbor_long[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(id, year, var_name)]

# Reshape back to wide format
stats_wide <- dcast(stats_dt, id + year ~ var_name,
                    value.var = c("max_val", "min_val", "mean_val"))

# Merge with original data
cell_dt <- merge(cell_dt, stats_wide, by = c("id", "year"), all.x = TRUE)

# Random Forest prediction (model preserved)
# Assume rf_model is loaded
preds <- predict(rf_model, newdata = cell_dt)
```

**Why This Works**  
- Eliminates per-row loops and string concatenation.  
- Uses `data.table` joins and aggregation for speed and memory efficiency.  
- Computes all neighbor stats in one grouped operation instead of multiple passes.  
- Scales to millions of rows without excessive copying.  

This approach should reduce runtime from **86+ hours to a few hours or less**, depending on disk I/O and CPU.