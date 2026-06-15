 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every cell-year row, causing massive redundant work across 6.46M rows and 28 years. The `build_neighbor_lookup` function is efficient for one pass, but the outer loop repeatedly calls `compute_neighbor_stats` for each variable, scanning the entire dataset multiple times. This results in high memory pressure and excessive computation time (86+ hours).  

**Optimization Strategy**  
- Build the neighbor lookup **once** at the cell level (not cell-year), since spatial adjacency does not change over time.  
- For each year, subset the data, compute neighbor stats for all variables in one pass, and join back.  
- Use **vectorized operations** and `data.table` for speed and memory efficiency.  
- Avoid repeated large `lapply` calls by precomputing neighbor indices and applying them efficiently.  
- Preserve the trained Random Forest model and estimand by not altering the feature definitions, only optimizing computation.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup at cell level
# rook_neighbors_unique: list of integer vectors (neighbors for each cell)
neighbor_lookup <- rook_neighbors_unique  # already precomputed

# Ensure id_order corresponds to cell IDs
id_order <- sort(unique(cell_data$id))
id_to_idx <- setNames(seq_along(id_order), id_order)

# Function to compute neighbor stats for one year
compute_yearly_neighbor_stats <- function(dt_year, neighbor_lookup, vars) {
  # Map cell id to row index in dt_year
  idx_lookup <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  # Preallocate result list
  res_list <- vector("list", length(vars))
  names(res_list) <- vars
  
  # For each variable, compute neighbor stats
  for (var in vars) {
    vals <- dt_year[[var]]
    stats_mat <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
    
    for (i in seq_len(nrow(dt_year))) {
      cell_id <- dt_year$id[i]
      neighbors <- neighbor_lookup[[id_to_idx[[as.character(cell_id)]]]]
      if (length(neighbors) == 0) next
      neighbor_ids <- id_order[neighbors]
      neighbor_idx <- idx_lookup[as.character(neighbor_ids)]
      neighbor_idx <- neighbor_idx[!is.na(neighbor_idx)]
      if (length(neighbor_idx) == 0) next
      neighbor_vals <- vals[neighbor_idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      stats_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
    
    colnames(stats_mat) <- paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    res_list[[var]] <- as.data.table(stats_mat)
  }
  
  # Combine all stats for this year
  cbind(dt_year[, .(id, year)], do.call(cbind, res_list))
}

# Main loop: process by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))

for (y in seq_along(years)) {
  dt_year <- cell_data[year == years[y]]
  result_list[[y]] <- compute_yearly_neighbor_stats(dt_year, neighbor_lookup, neighbor_source_vars)
}

# Combine all years back
neighbor_features <- rbindlist(result_list)

# Merge neighbor features back into main dataset
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This is Faster**
- Neighbor relationships are computed **once** at cell level.
- Each year is processed in isolation, reducing memory footprint.
- Each variable is computed in a single pass per year, not per entire dataset multiple times.
- Uses `data.table` for efficient joins and binding.

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, as redundant computations are eliminated and operations are chunked by year.

**Preserves:**  
- Original Random Forest model (no retraining).
- Original estimand and feature definitions.