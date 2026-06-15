 **Diagnosis**  
The current implementation is slow and memory-heavy because:  
1. **`lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small lists.  
2. **Repeated string concatenation and lookups**: `paste` and `setNames` operations for every row are expensive.  
3. **Non-vectorized neighbor aggregation**: Computing max/min/mean per row in R loops is inefficient.  
4. **Memory overhead**: Storing large lists in R consumes significant RAM.  

---

### **Optimization Strategy**
- **Avoid row-wise loops**: Use vectorized joins or matrix indexing instead of `lapply`.  
- **Precompute neighbor relationships once**: Expand neighbor pairs into a long table and join with values.  
- **Use `data.table`**: Efficient for large joins and aggregations.  
- **Chunk processing if needed**: To fit within 16 GB RAM.  
- **Preserve trained model and estimand**: Only feature engineering changes.  

---

### **Optimized Approach**
1. Convert `cell_data` to `data.table`.  
2. Create a **neighbor pair table**: `(cell_id, year, neighbor_id)`.  
3. Join neighbor values for each variable and compute `max`, `min`, `mean` by `(cell_id, year)`.  
4. Merge results back to `cell_data`.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs
# rook_neighbors_unique: list of neighbors per id_order index
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]
# Remove extra grouping
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year)]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  setnames(dt, var_name, "neighbor_val")
  
  # Aggregate stats by (id, year)
  stats <- dt[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle empty neighbor sets (replace Inf/-Inf with NA)
  stats[is.infinite(max_val), max_val := NA]
  stats[is.infinite(min_val), min_val := NA]
  
  # Merge back to main table
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(var_name)
}
```

---

### **Why This Is Faster**
- **Single join per variable** instead of millions of lookups.
- **Vectorized aggregation** with `data.table` instead of `lapply`.
- **Memory efficiency**: No massive nested lists.

---

**Expected Performance**:  
- From 86+ hours → likely under 1–2 hours on a 16 GB laptop (depending on disk speed).  
- If memory is tight, process variables one by one (as shown) or chunk by years.  

This preserves the original estimand and avoids retraining the model.