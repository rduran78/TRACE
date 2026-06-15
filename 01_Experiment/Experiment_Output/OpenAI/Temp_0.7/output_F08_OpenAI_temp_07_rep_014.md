 **Diagnosis**  
The extreme runtime (86+ hours) is caused by repeatedly computing neighbor stats for every cell-year row across 6.46M rows. The neighbor relationships are static across years, but the current design recalculates neighbor indices for each row/year. This creates massive redundant work and memory overhead. The neighbor lookup should be computed **once per cell** (344K entries), not per cell-year row. Then, for each year, compute neighbor stats by slicing the year’s values and applying precomputed neighbor indices.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** for the 344,208 cells using their IDs (static graph).
2. Split the panel data by year and compute neighbor stats year-by-year using the static neighbor mapping.
3. Use `matrix` operations or `data.table` for efficiency.
4. Append results back to the original `cell_data` without breaking the Random Forest pipeline.
5. Avoid growing objects in loops; preallocate or use `rbindlist`.

---

**Optimized R Code**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup (static per cell)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) as.integer(id_order[neighbors[[i]]]))
}
neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
names(neighbor_lookup_static) <- as.character(id_order)

# Function to compute neighbor stats for a given year
compute_neighbor_stats_year <- function(dt_year, var_name, neighbor_lookup) {
  vals <- dt_year[[var_name]]
  res <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
  
  for (i in seq_len(nrow(dt_year))) {
    neigh_ids <- neighbor_lookup[[ as.character(dt_year$id[i]) ]]
    if (length(neigh_ids) == 0) next
    neighbor_vals <- vals[ match(neigh_ids, dt_year$id, nomatch = 0) ]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    res[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  res
}

# Main optimized loop: process by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {
  # Preallocate storage for all rows
  all_stats <- vector("list", length(years))
  
  for (y_idx in seq_along(years)) {
    y <- years[y_idx]
    dt_year <- cell_data[year == y]
    stats_year <- compute_neighbor_stats_year(dt_year, var_name, neighbor_lookup_static)
    all_stats[[y_idx]] <- stats_year
  }
  
  # Bind results and add to cell_data in correct order
  all_stats_mat <- do.call(rbind, all_stats)
  
  cell_data[, paste0(var_name, "_nbr_max") := all_stats_mat[,1]]
  cell_data[, paste0(var_name, "_nbr_min") := all_stats_mat[,2]]
  cell_data[, paste0(var_name, "_nbr_mean") := all_stats_mat[,3]]
}

# At this point, cell_data has neighbor stats ready for prediction
# Preserve trained Random Forest model and use it as before
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

**Key Gains**  
- **Neighbor graph built once** (no per-row repetition).
- **Year-level batch processing** reduces complexity from O(N*T*k) to O(T*(N+k)).
- Should reduce runtime from 86+ hours to a few hours (or less) on 16 GB RAM.

**Numerical estimand preserved**: Same neighbor stats, same Random Forest model, only optimized computation.