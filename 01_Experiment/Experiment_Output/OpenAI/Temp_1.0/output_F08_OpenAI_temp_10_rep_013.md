 **Diagnosis**  
The bottleneck is that computation in `compute_neighbor_stats` loops over ~6.46 million rows for each variable, performing nested lookups into large lists. Since the neighbor structure is static across years, recalculating neighbor indices per row-year results in redundant work. The current O(N × Y × neighbors) approach explodes in size, leading to the 86+ hour estimate and memory pressure.

**Optimization Strategy**  
- Precompute a static neighbor *index matrix* based only on cell ids (size ~344k rows).
- For each year, slice the relevant variable vector, use the static neighbor indices, and compute max/min/mean in a vectorized way.
- Process year-wise in memory-efficient batches rather than across all rows.
- Append features back to the panel after computing year-specific neighbor stats.
- Avoid repeated `lapply` and `do.call` by using matrix operations or vectorized apply.

This reduces complexity from ~6.46M × neighbors per variable repeated for every variable → down to 28 batches with ~344k computations each.

---

### **Optimized Working R Code**

```r
# Build static neighbor matrix once based on cell_id
build_static_neighbor_matrix <- function(id_order, neighbors) {
  # neighbors is spdep::nb for id_order
  max_nbrs <- max(sapply(neighbors, length))
  # Fill matrix with NA for missing slots
  nbr_mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_nbrs)
  for (i in seq_along(neighbors)) {
    if (length(neighbors[[i]]) > 0) {
      nbr_mat[i, seq_along(neighbors[[i]])] <- neighbors[[i]]
    }
  }
  nbr_mat
}

compute_year_neighbor_stats <- function(year_data, nbr_mat, var_name) {
  vals <- year_data[[var_name]]
  # Extract neighbors' values in matrix form
  neighbor_vals <- apply(nbr_mat, 2, function(col_idx) {
    v <- vals[col_idx]
    v[is.na(col_idx)] <- NA_real_
    v
  })
  neighbor_vals <- as.matrix(neighbor_vals)
  # Compute row-wise max, min, mean with NA handling
  max_vals <- apply(neighbor_vals, 1, function(x) if(all(is.na(x))) NA_real_ else max(x, na.rm=TRUE))
  min_vals <- apply(neighbor_vals, 1, function(x) if(all(is.na(x))) NA_real_ else min(x, na.rm=TRUE))
  mean_vals <- apply(neighbor_vals, 1, function(x) if(all(is.na(x))) NA_real_ else mean(x, na.rm=TRUE))
  cbind(max_vals, min_vals, mean_vals)
}

# Main process
optimize_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_vars) {
  nbr_mat <- build_static_neighbor_matrix(id_order, neighbors)
  
  results_list <- vector("list", length(neighbor_vars))
  names(results_list) <- neighbor_vars
  
  # Preallocate for final combination
  cell_data_out <- cell_data
  
  for (yr in sort(unique(cell_data$year))) {
    year_idx <- which(cell_data$year == yr)
    year_data <- cell_data[year_idx, ]
    # Order year_data by id_order if necessary
    year_data <- year_data[match(id_order, year_data$id), ]
    
    # Compute stats for each neighbor var
    for (var_name in neighbor_vars) {
      stats_mat <- compute_year_neighbor_stats(year_data, nbr_mat, var_name)
      colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
      # Merge back by position
      cell_data_out[year_idx, paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))] <- stats_mat
    }
  }
  
  cell_data_out
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

**Why this works**  
- Neighbor lookup (`nbr_mat`) is computed once, size ~344k × max degree.
- Processing is per-year, so only 344k rows in memory at a time. For 28 loop iterations and 5 variables it's practical on 16 GB RAM.
- Feature consistency preserved (max, min, mean).
- No model retraining; we enrich `cell_data` exactly as before.

**Expected performance improvement**  
From 86+ hours → likely under 1-2 hours depending on disk and CPU, due to eliminating 6.46M deep list traversals and redundant concatenations.