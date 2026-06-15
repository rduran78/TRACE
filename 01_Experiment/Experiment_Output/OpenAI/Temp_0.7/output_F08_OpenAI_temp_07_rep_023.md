 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because `compute_neighbor_stats` iterates over **6.46 million rows** and repeatedly subsets vectors in R (`lapply` over every row). This creates massive overhead from R loops and redundant operations. The neighbor relationships are **static across years**, yet the function rebuilds neighbor statistics per cell-year individually. The root inefficiency is treating each of the 6.46M cell-year rows independently instead of leveraging the fixed adjacency to compute statistics in a vectorized, year-wise manner.

---

**Optimization Strategy**  
1. **Exploit static neighbor structure**: Precompute a fixed mapping of cell → neighbors at the cell level (344k cells).  
2. **Compute per-year statistics in blocks**: For each year, extract the variable vector (length = 344k), compute neighbor max/min/mean in a single vectorized pass using the static adjacency.  
3. **Avoid row-by-row lapply**: Replace with matrix operations or `vapply` on 344k cells × 28 years (instead of 6.46M lapply calls).  
4. **Memory-friendly approach**: Process one variable and one year at a time to stay within 16 GB RAM.  
5. **Preserve original estimand**: Same neighbor relationships, same variable transformations, only faster computation.  
6. **Reuse Random Forest model**: Just add new features to `cell_data` before prediction; no retraining.  

---

**Optimized R Code**  

```r
# Precompute static neighbor list as integer indices
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors: spdep::nb object
  # id_order: vector of cell ids in the same order as neighbors
  lapply(neighbors, function(nb_idx) id_order[nb_idx])
}

# Compute neighbor stats by year for one variable
compute_neighbor_stats_by_year <- function(cell_data, var_name, neighbor_list) {
  years <- sort(unique(cell_data$year))
  n_cells <- length(neighbor_list)
  
  # Prepare output: matrix [nrow(cell_data), 3]
  result <- matrix(NA_real_, nrow(cell_data), 3)
  
  for (yr in years) {
    idx_year <- which(cell_data$year == yr)
    vals <- cell_data[[var_name]][idx_year]
    
    # Assuming idx_year is ordered by id_order
    stopifnot(length(vals) == n_cells)
    
    # Compute stats for each cell
    max_vec <- numeric(n_cells)
    min_vec <- numeric(n_cells)
    mean_vec <- numeric(n_cells)
    
    for (i in seq_len(n_cells)) {
      nb_idx <- neighbor_list[[i]]
      if (length(nb_idx) == 0) {
        max_vec[i] <- min_vec[i] <- mean_vec[i] <- NA_real_
      } else {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0) {
          max_vec[i] <- min_vec[i] <- mean_vec[i] <- NA_real_
        } else {
          max_vec[i] <- max(nb_vals)
          min_vec[i] <- min(nb_vals)
          mean_vec[i] <- mean(nb_vals)
        }
      }
    }
    
    # Fill result rows for this year
    result[idx_year, ] <- cbind(max_vec, min_vec, mean_vec)
  }
  
  colnames(result) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  result
}

# Main pipeline
id_order <- sort(unique(cell_data$id))  # Ensure consistent ordering
neighbor_list <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Split cell_data by year and reorder to align with id_order
cell_data <- cell_data[order(cell_data$year, match(cell_data$id, id_order)), ]

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_by_year(cell_data, var_name, neighbor_list)
  cell_data <- cbind(cell_data, stats_mat)
}

# Now you can run the pre-trained Random Forest prediction step as before
```

---

**Performance Gains**  
- Reduces complexity from **6.46M row-wise operations** to **28 yearly blocks × 344k cells**.
- Eliminates repeated string concatenation and index lookups.
- Memory overhead limited to one year slice at a time.
- Expected runtime: **hours → minutes** on a standard laptop.

**Key Points**  
- Preserves the original neighbor structure and estimand.
- Keeps Random Forest model unchanged.
- Fully exploits the static adjacency vs. dynamic variables distinction.