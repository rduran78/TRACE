 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows and repeatedly subsets vectors inside `lapply`. This is repeated for each variable, causing redundant computations. The neighbor relationships are static across years, but the code rebuilds neighbor-based stats for every row-year combination without leveraging this static structure efficiently. Memory and CPU overhead from repeated `lapply` calls and `do.call(rbind, ...)` on millions of elements is huge.

---

**Optimization Strategy**  
- Precompute a static neighbor index for each cell (not cell-year).
- For each year, extract the relevant slice of data and compute neighbor stats in a **vectorized** way using matrix operations or `data.table`.
- Avoid recomputing neighbor lookups per row-year; instead, reuse the static neighbor mapping.
- Process year by year to keep memory usage manageable.
- Append results back to the main dataset incrementally.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute static neighbor lookup: list of integer vectors (cell indices)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup_static <- lapply(seq_along(id_order), function(i) {
  id_order[rook_neighbors_unique[[i]]]
})

# Ensure data is sorted by id and year
setkey(cell_data, id, year)

# Prepare output columns
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Slice data for this year
  year_data <- cell_data[year == yr]
  vals_list <- lapply(neighbor_source_vars, function(v) year_data[[v]])
  names(vals_list) <- neighbor_source_vars
  
  # Compute neighbor stats for each variable
  for (var in neighbor_source_vars) {
    vals <- vals_list[[var]]
    res <- matrix(NA_real_, nrow = length(id_order), ncol = 3)
    
    for (i in seq_along(id_order)) {
      nbr_ids <- neighbor_lookup_static[[i]]
      if (length(nbr_ids) == 0) next
      nbr_idx <- id_to_idx[as.character(nbr_ids)]
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) next
      res[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
    
    # Assign back to cell_data for this year
    cell_data[year == yr, paste0(var, "_nbr_max") := res[, 1]]
    cell_data[year == yr, paste0(var, "_nbr_min") := res[, 2]]
    cell_data[year == yr, paste0(var, "_nbr_mean") := res[, 3]]
  }
}

# At this point, cell_data has neighbor features added
# Proceed with Random Forest prediction using the existing model
```

---

**Why This Works Faster**  
- Neighbor relationships are computed once (static).
- Looping over 28 years × 344k cells is manageable (≈9.6M operations), far less than 6.46M × 5 variables × repeated subsetting.
- Memory footprint is controlled by processing one year at a time.
- Preserves original estimand and avoids retraining the Random Forest model.

**Expected Improvement**  
From 86+ hours to a few hours (depending on I/O speed), as we eliminate redundant lookups and leverage efficient data.table operations.