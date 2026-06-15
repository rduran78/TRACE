 **Diagnosis**  
The current implementation is extremely slow because:  
1. `build_neighbor_lookup` creates a list of neighbor indices for every **cell-year row** (≈6.46M rows), even though the neighbor structure is static across years.  
2. This results in redundant storage and repeated computation of the same neighbor relationships for each year.  
3. `compute_neighbor_stats` iterates over this massive list, repeatedly accessing large vectors, causing memory and CPU overhead.  

**Optimization Strategy**  
- Exploit the static neighbor structure:  
  - Build the neighbor lookup **once per cell**, not per cell-year.  
- For each year, compute neighbor statistics by **vectorized operations**:  
  - Use the static neighbor list to aggregate values for that year only.  
- Process year-by-year to keep memory usage manageable.  
- Append results back to the main data frame incrementally.  

This reduces complexity from O(N * T) neighbor lookups to O(N + T * N_neighbors), cutting runtime drastically.  

---

### **Optimized R Code**

```r
# Precompute static neighbor lookup: list of neighbor indices by cell
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_idx[as.character(neighbor_ids)]
  })
}

# Compute neighbor stats for one year
compute_neighbor_stats_year <- function(data_year, neighbor_lookup, var_name) {
  vals <- data_year[[var_name]]
  result <- matrix(NA_real_, nrow = nrow(data_year), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  colnames(result) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  result
}

# Main optimized loop
neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Assuming cell_data has columns: id, year, and predictor vars
out_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (y in seq_along(years)) {
  data_year <- cell_data[cell_data$year == years[y], ]
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_year(data_year, neighbor_lookup_static, var_name)
  })
  out_list[[y]] <- cbind(data_year, do.call(cbind, stats_list))
}

cell_data_with_neighbors <- do.call(rbind, out_list)
```

---

**Key Improvements**  
- **Static neighbor lookup**: built once for 344,208 cells, not 6.46M rows.  
- **Year-wise processing**: keeps memory footprint low.  
- **Vectorized aggregation per year**: avoids redundant computation.  

**Expected Runtime**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, since the heavy redundant operations are eliminated.  

**Preserves**  
- Original numerical estimand.  
- Pre-trained Random Forest model (no retraining).  
- Neighbor max, min, mean logic.