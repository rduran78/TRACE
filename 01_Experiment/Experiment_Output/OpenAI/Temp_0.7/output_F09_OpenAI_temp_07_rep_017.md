 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every variable and year, repeatedly scanning a 6.46M-row dataset. This is highly inefficient because neighbor-cell relationships are static across years. The `build_neighbor_lookup` function creates an index for all rows, but the outer loop repeatedly calls `compute_and_add_neighbor_features` per variable, leading to redundant lookups and excessive allocations. With 6.46M rows, 28 years, and 5 variables, this multiplies the workload dramatically.  

**Optimization Strategy**  
- Build a single reusable adjacency table (`neighbor_table`) mapping each row to its neighbors once.  
- Perform a single join per year:  
  1. Attach yearly variable values to the neighbor table.  
  2. Compute neighbor stats (max, min, mean) using vectorized operations or `data.table`.  
  3. Join results back to the main dataset.  
- Leverage `data.table` for efficient grouping and aggregation.  
- Keep memory usage bounded by processing one year at a time.  
- Preserve the trained Random Forest model and original target.  

**Optimized R Code**

```r
library(data.table)

# Assume: cell_data (id, year, variables...), rook_neighbors_unique, id_order exist
setDT(cell_data)

# Build reusable adjacency table (cell_id -> neighbor_id)
neighbor_table <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Repeat each year for full panel
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure keys for fast joins
setkey(cell_data, id, year)

for (yr in years) {
  message("Processing year: ", yr)
  
  # Subset data for this year
  year_data <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_data, id)
  
  # Join neighbor_table with year_data to get neighbor values
  nb_vals <- neighbor_table[year_data, on = .(neighbor_id = id)]
  # nb_vals now has: id (cell), neighbor_id, year, vars...
  
  # Compute stats for each variable
  for (var in neighbor_source_vars) {
    stats <- nb_vals[!is.na(get(var)), .(
      n_max = max(get(var), na.rm = TRUE),
      n_min = min(get(var), na.rm = TRUE),
      n_mean = mean(get(var), na.rm = TRUE)
    ), by = .(id, year)]
    
    # Merge stats back into cell_data
    setnames(stats, c("n_max", "n_min", "n_mean"),
             paste0(var, c("_nb_max", "_nb_min", "_nb_mean")))
    cell_data[year == yr, (names(stats)[-c(1,2)]) := stats[, -c("id","year"), with = FALSE], on = .(id, year)]
  }
}

# At this point, cell_data has neighbor-based features for all years and variables.
# Proceed with prediction using the preserved Random Forest model:
predictions <- predict(rf_model, newdata = cell_data)
```

**Why this is faster:**  
- Neighbor relationships computed once and reused.  
- Aggregations are vectorized via `data.table` instead of per-row `lapply`.  
- Memory load is controlled by processing one year at a time.  

This approach reduces redundant computations and should cut runtime from 86+ hours to a few hours, depending on I/O and CPU performance.