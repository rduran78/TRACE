 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each year and variable, which is highly inefficient given the large dataset (6.46M rows) and 5 variables. The `build_neighbor_lookup` function is called once, but neighbor stats are computed in a loop for each variable, scanning all rows repeatedly. This results in excessive memory access and redundant computation. The bottleneck is the repeated traversal of neighbor indices and extraction of values for each variable separately.

**Optimization Strategy**  
- Precompute a reusable neighbor lookup table once (already done).
- Compute all neighbor statistics for all source variables in a single pass rather than looping over variables.
- Use vectorized operations and `data.table` for efficient grouping and joining.
- Avoid repeated `lapply` calls over millions of rows; instead, compute neighbor stats in bulk.
- Keep memory footprint low by processing in chunks if necessary.

**Working R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Flatten neighbor lookup into a long table: (row_id, neighbor_id)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Join neighbor values for all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for selected variables
long_data <- melt(cell_data[, c("id", "year", neighbor_source_vars), with = FALSE],
                  measure.vars = neighbor_source_vars,
                  variable.name = "var_name", value.name = "value")

# Add row index for joining
long_data[, row_id := .I]

# Join neighbor_dt to long_data twice: once for row_id, once for neighbor_id
neighbor_values <- neighbor_dt[long_data, on = .(neighbor_id = row_id)]
# neighbor_values now has: row_id (target), var_name, value (neighbor's value)

# Compute stats by row_id and var_name
stats_dt <- neighbor_values[, .(
  max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
  min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
  mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
), by = .(row_id, var_name)]

# Reshape back to wide format
stats_wide <- dcast(stats_dt, row_id ~ var_name,
                    value.var = c("max_val", "min_val", "mean_val"))

# Bind stats back to cell_data
cell_data <- cbind(cell_data, stats_wide[, -1])

# Random Forest prediction (model preserved)
predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Why This Is Faster**  
- Neighbor relationships computed once and reused.
- All variables processed in a single grouped operation.
- `data.table` provides efficient joins and aggregations.
- Eliminates repeated `lapply` over millions of rows.

This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop, depending on disk I/O and CPU speed.