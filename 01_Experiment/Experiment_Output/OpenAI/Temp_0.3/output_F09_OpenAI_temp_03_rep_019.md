 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each year and variable, causing massive redundant work. With ~6.46M rows and 5 variables, iterating through all rows and re-deriving neighbor indices is extremely expensive. The bottleneck is the repeated neighbor lookup and aggregation across 28 years × 344k cells.  

**Optimization Strategy**  
- Build a **single reusable neighbor lookup** based on cell IDs (not cell-years).  
- For each year, **join yearly attributes** to that lookup and compute neighbor stats in a **vectorized manner**.  
- Avoid repeated `lapply` over millions of rows; instead, use `data.table` for fast grouping and joins.  
- Compute all neighbor stats in one pass per variable per year.  
- Preserve the trained Random Forest model and estimand by keeping feature definitions identical.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute reusable neighbor table (cell-to-cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Keys for fast joins
setkey(neighbor_dt, neighbor_id)
setkey(cell_data, id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_year <- function(var_name) {
  # For each year, join and aggregate
  result_list <- lapply(unique(cell_data$year), function(yr) {
    # Subset for this year
    year_data <- cell_data[year == yr, .(id, val = get(var_name))]
    
    # Join neighbor values
    joined <- neighbor_dt[year_data, on = .(neighbor_id = id)]
    
    # Aggregate neighbor stats per focal id
    stats <- joined[, .(
      n_max = max(val, na.rm = TRUE),
      n_min = min(val, na.rm = TRUE),
      n_mean = mean(val, na.rm = TRUE)
    ), by = id]
    
    # Merge back to main data
    stats[, year := yr]
    setnames(stats, c("n_max", "n_min", "n_mean"),
             paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
    stats
  })
  
  rbindlist(result_list)
}

# Compute for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_stats <- lapply(neighbor_source_vars, compute_neighbor_stats_year)

# Merge all stats into cell_data
for (stats in all_stats) {
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# At this point, cell_data has neighbor features ready for Random Forest prediction
# Predict using existing trained model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

### **Why This is Faster**
- **Neighbor relationships computed once** (1.37M rows vs 6.46M × 5 loops).
- **Yearly joins** avoid recomputing neighbor indices.
- **Vectorized aggregation** via `data.table` replaces millions of `lapply` calls.
- Memory efficient: works year-by-year instead of loading all combinations.

**Expected improvement:** From 86+ hours to a few hours (or less) on a 16 GB laptop.