 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every year-variable combination, causing massive redundant lookups across ~6.46M rows. Since spatial adjacency is static, rebuilding neighbor indices repeatedly is unnecessary. The main bottleneck is the repeated `lapply` over millions of rows for each variable, combined with dynamic string-based lookups.  

**Optimization Strategy**  
- Precompute a reusable neighbor lookup **once** at the cell level (not cell-year).  
- For each year, join yearly attributes onto this lookup and compute neighbor stats in a **vectorized** way using `data.table` or `dplyr`.  
- Avoid repeated string concatenation and repeated `lapply` calls.  
- Use matrix operations or grouped joins for speed.  
- Preserve the trained Random Forest model and estimand by only changing feature engineering.  

---

### **Optimized R Implementation**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup at cell level
# rook_neighbors_unique: list of integer vectors (neighbors)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Set keys for fast joins
setkey(neighbor_dt, neighbor_id)
setkey(cell_data, id, year)

# Function to compute neighbor stats for a single variable
compute_neighbor_stats_fast <- function(var_name) {
  # Join neighbor values for each year
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(tmp, neighbor_id, year)
  
  joined <- neighbor_dt[tmp, allow.cartesian = TRUE]  # join neighbors
  # joined now has: id (focal), neighbor_id, year, val
  
  # Aggregate neighbor stats by focal id and year
  stats <- joined[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle cases where all neighbors are NA
  stats[!is.finite(max_val), `:=`(max_val = NA, min_val = NA, mean_val = NA)]
  
  # Merge back to main data
  setnames(stats, c("id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  cell_data <- compute_neighbor_stats_fast(var_name)
}

# At this point, cell_data has new neighbor features
# Proceed with prediction using existing Random Forest model
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This is Faster**
- **Neighbor lookup built once** (O(n_neighbors) ~ 1.37M rows).
- **Vectorized joins** replace millions of `lapply` calls.
- **Single pass per variable** instead of per cell-year.
- Memory efficient: uses `data.table` joins instead of repeated list operations.

**Expected runtime reduction:** From 86+ hours to a few hours (depending on I/O), well within laptop limits.