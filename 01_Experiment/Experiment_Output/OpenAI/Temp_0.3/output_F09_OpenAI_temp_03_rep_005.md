 **Diagnosis**  
The current pipeline is extremely slow because for ~6.46M rows, `compute_neighbor_stats` repeatedly scans neighbor indices and extracts values from the full dataset for each row. This results in heavy repeated lookups and memory thrashing. The neighbor structure is static across years, but the code rebuilds neighbor relationships conceptually for every row. Additionally, the `lapply` approach is not vectorized and scales poorly.

**Optimization Strategy**  
1. **Precompute and reuse adjacency**: Build a reusable neighbor index table once at the cell level, not per row.  
2. **Split by year**: For each year, join yearly attributes to the static neighbor table and compute neighbor stats in a vectorized way.  
3. **Use `data.table` for speed**: Perform joins and aggregations efficiently.  
4. **Avoid repeated R loops**: Compute all neighbor stats for all variables in a single pass per year.  
5. **Preserve trained model and estimand**: Only speed up feature engineering; do not retrain the Random Forest.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (cell-level)
# rook_neighbors_unique: list of integer vectors
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Ensure keys for fast joins
setkey(neighbor_pairs, neighbor_id)
setkey(cell_data, id, year)

# Variables for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result container
result_list <- vector("list", length = length(unique(cell_data$year)))

# Compute neighbor stats per year
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for this year
  year_data <- cell_data[year == yr, .(id, year, (neighbor_source_vars)), with = FALSE]
  setkey(year_data, id)
  
  # Join neighbor attributes
  joined <- neighbor_pairs[year_data, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined: id (cell), neighbor_id, ntl, ec, pop_density, def, usd_est_n2
  
  # Compute stats by original cell id
  agg <- joined[, lapply(.SD, function(x) {
    if (all(is.na(x))) return(c(NA_real_, NA_real_, NA_real_))
    c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))
  }), by = id, .SDcols = neighbor_source_vars]
  
  # Reshape columns: var_max, var_min, var_mean
  new_cols <- unlist(lapply(neighbor_source_vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  agg <- data.table(id = agg$id, matrix(unlist(agg[, -1, with = FALSE]), ncol = length(new_cols), byrow = TRUE))
  setnames(agg, c("id", new_cols))
  
  # Merge back to year_data
  year_result <- merge(year_data, agg, by = "id", all.x = TRUE)
  result_list[[as.character(yr)]] <- year_result
}

# Combine all years
cell_data_enriched <- rbindlist(result_list)

# Now you can run the Random Forest prediction using the enriched dataset
# rf_predictions <- predict(trained_rf_model, newdata = cell_data_enriched)
```

---

**Why this is faster**  
- **Neighbor table built once** (1.37M pairs vs. 6.46M rows × multiple lookups).  
- **Vectorized aggregation** using `data.table` instead of millions of `lapply` calls.  
- **Per-year processing** avoids unnecessary full-table joins.  
- **Memory efficient**: Works in chunks (by year) on a 16 GB laptop.

This approach should reduce runtime from **86+ hours to a few hours** depending on disk I/O and CPU.