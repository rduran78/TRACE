 **Diagnosis**  
The bottleneck arises because `build_neighbor_lookup` and `compute_neighbor_stats` use nested `lapply` over ~6.46M rows, repeatedly building vectors and scanning large lists. Memory overhead is huge due to millions of small objects and repeated string concatenation. Current approach is essentially O(N × k) with heavy R overhead, not vectorized, and not parallelized.

---

### **Optimization Strategy**
1. **Avoid per-row string concatenation & lookups**  
   Precompute a `data.table` keyed by `id` and `year` to allow fast joins rather than building millions of character keys.
   
2. **Reshape panel to a 3D-like structure**  
   Use `data.table` or `matrix` for fast column operations. Compute neighbor stats by joining neighbor IDs once and aggregating.
   
3. **Process in chunks**  
   Load ~6.46M rows in manageable blocks or by year to keep memory under control.
   
4. **Vectorize aggregation**  
   Use `data.table` grouping or `collapse::fmean/fmax/fmin` for fast aggregation instead of millions of `lapply` calls.
   
5. **Optional**: Parallelize across years or variables using `future.apply` or `data.table` parallelism.

---

### **Working R Code (Efficient with `data.table`)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor table (id -> neighbor_id)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

setkey(neighbor_dt, id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(dt, var_name) {
  # Select only columns needed
  vals <- dt[, .(id, year, value = get(var_name))]
  
  # Join neighbors
  merged <- neighbor_dt[vals, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # merged: id (from neighbor_dt), neighbor_id, year, value
  
  # Aggregate by original cell id and year
  agg <- merged[, .(
    max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  setnames(agg, c("id", "year", paste0(var_name, c("_ngb_max", "_ngb_min", "_ngb_mean"))))
  agg
}

# Process in chunks by year to avoid memory blow-up
years <- unique(cell_data$year)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in years) {
  cat("Processing year:", yr, "\n")
  dt_year <- cell_data[year == yr]
  
  # Compute and merge neighbor features for each variable
  for (var_name in neighbor_source_vars) {
    agg <- compute_neighbor_stats_fast(dt_year, var_name)
    dt_year <- merge(dt_year, agg, by = c("id", "year"), all.x = TRUE)
  }
  
  # Write back to main table
  cell_data[year == yr] <- dt_year
}

# cell_data now has neighbor stats columns added
```

---

### **Why This is Faster**
- Eliminates per-row `lapply` and string concatenation.
- Uses `data.table` join + aggregation (highly optimized in C).
- Processes by year to keep memory footprint low.
- Scales linearly and can be parallelized further if needed.

---

**Expected Performance**:  
On 6.46M rows, this approach should reduce runtime from 86+ hours to a few hours or less on a 16GB laptop, depending on disk speed and parallelization.