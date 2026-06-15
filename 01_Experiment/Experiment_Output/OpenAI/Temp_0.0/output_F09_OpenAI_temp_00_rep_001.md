 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each year and variable, which is highly inefficient given the large dataset (6.46M rows) and 5 neighbor-based variables. The `build_neighbor_lookup` function is called once, but neighbor stats are computed in a loop for each variable, scanning large lists repeatedly. This results in excessive memory usage and redundant computation. The bottleneck is the repeated traversal of neighbor indices and extraction of values for each variable across millions of rows.

**Optimization Strategy**  
- Precompute a reusable neighbor lookup table once (already done).
- Use vectorized operations or `data.table` joins instead of repeated `lapply` calls.
- Reshape data to long format for efficient joins.
- Compute neighbor statistics in a single pass per variable using aggregation rather than looping.
- Avoid copying large data frames repeatedly; add columns in place.
- Use `data.table` for memory efficiency and speed.

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Create a long table of (row_id, neighbor_id)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Function to compute and add neighbor stats for a variable
compute_and_add_neighbor_features_dt <- function(dt, var_name, neighbor_dt) {
  vals <- dt[[var_name]]
  # Join neighbor values
  neighbor_dt[, val := vals[neighbor_id]]
  # Aggregate stats by row_id
  stats <- neighbor_dt[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = row_id]
  # Add to original data.table
  dt[stats$row_id, paste0(var_name, "_nbr_max") := stats$max_val]
  dt[stats$row_id, paste0(var_name, "_nbr_min") := stats$min_val]
  dt[stats$row_id, paste0(var_name, "_nbr_mean") := stats$mean_val]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_dt(cell_data, var_name, neighbor_dt)
}

# Preserve trained Random Forest model and original estimand
# Predict using the existing model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Why This Is Faster**  
- Neighbor relationships are stored once in `neighbor_dt`.
- Aggregation uses `data.table` grouping, which is highly optimized in C.
- No repeated list traversals for each variable.
- Avoids copying large data frames repeatedly.

**Expected Performance Gain**  
This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop, depending on disk I/O and CPU speed.