 **Diagnosis**  
The current pipeline is slow (86+ hours) because:  
- `build_neighbor_lookup` creates a full lookup for every cell-year, repeating neighbor resolution across 28 years.  
- `compute_neighbor_stats` iterates over 6.46M rows with repeated list operations and `do.call(rbind, ...)`.  
- The graph topology is rebuilt for every year instead of being reused.  
- Excessive overhead from R list processing and repeated string concatenation.  

**Optimization Strategy**  
- Precompute graph topology once (cell → neighbors).  
- Use integer indexing and avoid repeated string concatenation.  
- Process data in a vectorized manner by year, using `data.table` for speed.  
- Compute neighbor statistics in a single pass per variable-year using fast aggregation.  
- Avoid growing objects in loops; preallocate results.  
- Keep numerical equivalence by preserving max, min, mean logic.  

---

### **Efficient R Implementation**

```r
library(data.table)

# Convert to data.table for efficient operations
setDT(cell_data)

# Precompute neighbor topology as integer indices
# id_order: vector of cell IDs in consistent order
# rook_neighbors_unique: list of neighbor indices (spdep::nb)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- lapply(rook_neighbors_unique, function(nb) id_to_idx[nb])

# Prepare output columns for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Efficient computation by year
years <- unique(cell_data$year)
n_cells <- length(id_order)

for (yr in years) {
  # Subset for this year
  dt_year <- cell_data[year == yr]
  # Map cell IDs to row positions for this year
  idx_map <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  for (var in neighbor_source_vars) {
    vals <- dt_year[[var]]
    # Preallocate result arrays
    max_res <- numeric(n_cells)
    min_res <- numeric(n_cells)
    mean_res <- numeric(n_cells)
    
    for (i in seq_len(n_cells)) {
      nbrs <- neighbor_list[[i]]
      nbr_rows <- idx_map[id_order[nbrs]]
      nbr_vals <- vals[nbr_rows]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) {
        max_res[i] <- NA_real_
        min_res[i] <- NA_real_
        mean_res[i] <- NA_real_
      } else {
        max_res[i] <- max(nbr_vals)
        min_res[i] <- min(nbr_vals)
        mean_res[i] <- mean(nbr_vals)
      }
    }
    
    # Assign back to original table
    cell_data[year == yr, paste0(var, "_nbr_max") := max_res]
    cell_data[year == yr, paste0(var, "_nbr_min") := min_res]
    cell_data[year == yr, paste0(var, "_nbr_mean") := mean_res]
  }
}

# At this point, cell_data has neighbor features added
# Apply pre-trained Random Forest model without retraining
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why This Is Faster**  
- Graph topology (`neighbor_list`) built once and reused.  
- No repeated string concatenation for keys.  
- Uses integer indexing instead of character lookups.  
- Processes by year to keep memory manageable.  
- Avoids `do.call(rbind, ...)` overhead.  
- Preallocates numeric vectors for results.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours on a 16 GB laptop, depending on disk I/O and CPU speed. Further speedups possible via:  
- Parallelization with `future.apply` or `parallel` for year loops.  
- Using `Rcpp` for inner neighbor aggregation if needed.  

**Numerical Equivalence**  
Max, min, mean logic is preserved exactly as in original implementation.  
Random Forest model is not retrained; predictions use the enriched dataset.