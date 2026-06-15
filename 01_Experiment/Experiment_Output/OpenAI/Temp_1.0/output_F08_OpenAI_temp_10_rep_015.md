 **Diagnosis**  
The slowness (86+ hours) is due to computing neighbor statistics repeatedly across 6.46 M rows, even though neighbor relationships are static. Each call to `compute_neighbor_stats` performs multiple traversals of large vectors, leading to redundant work. Memory overhead compounds with dynamic indexing.  

Key inefficient points:  
- For each row-year, neighbor indices are recomputed in terms of the repeated variable subset.  
- Entire pipeline treats each row (cell-year) independently—a 6.46 M × 5 multi-pass.  
- Year looping is implicit in repeated neighbor-stat computation rather than isolated at column-level.  

Since neighbor relationships never change over time, we should:  
- Compute neighbor adjacencies once for the 344,208 cells (static).  
- For each year, slice the relevant variable vector, compute neighbor stats for all cells in one shot, and append results.  
- Avoid creating gigantic per-row lists of neighbors for every cell–year (currently 6.46 M lists).  

---

**Optimization Strategy**  
1. **Precompute adjacency**: Keep neighbor lookup at cell (not cell-year) level.  
2. **Process by year**: For each year and each variable, use the static adjacency to compute max, min, mean.  
3. **Vectorize**: Use matrix operations or `vapply` instead of repeatedly building row-level lists.  
4. **Incremental write**: Append results per year instead of holding everything in memory at once.  

Benefit: We cut complexity from \(O(N_\text{rows} \times V)\) to ~\(O(N_\text{cells} \times Y \times V)\), reusing adjacency each time.  

---

**Optimized R Code**

```r
# Static neighbor indices (cell-level)
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors: spdep::nb list of integer indices
  # id_order is vector of cell IDs in order
  out <- lapply(seq_along(id_order), function(i) neighbors[[i]])
  names(out) <- id_order
  out
}

# Compute a matrix of neighbor stats for one year and one variable
compute_neighbor_stats_year <- function(var_values, neighbor_lookup) {
  result <- t(vapply(neighbor_lookup, function(idx) {
    neigh_vals <- var_values[idx]
    neigh_vals <- neigh_vals[!is.na(neigh_vals)]
    if (length(neigh_vals) == 0) return(c(NA, NA, NA))
    c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
  }, numeric(3)))
  colnames(result) <- c("max", "min", "mean")
  result
}

# Main optimized loop
add_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  neighbor_lookup <- build_cell_neighbor_lookup(id_order, neighbors)
  years <- sort(unique(cell_data$year))
  
  # Preallocate a list to collect augmented data per year
  augmented_list <- vector("list", length(years))
  
  for (y_idx in seq_along(years)) {
    yr <- years[y_idx]
    message("Processing year: ", yr)
    
    # Filter once for the year
    year_subset <- cell_data[cell_data$year == yr, ]
    stopifnot(nrow(year_subset) == length(id_order)) # ensure alignment
    
    # For each variable, compute stats
    for (var_name in vars) {
      stats_mat <- compute_neighbor_stats_year(year_subset[[var_name]], neighbor_lookup)
      year_subset[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
      year_subset[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
      year_subset[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
    }
    
    augmented_list[[y_idx]] <- year_subset
  }
  
  do.call(rbind, augmented_list)
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data_enriched <- add_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)
```

---

**Why This Works**  
- `neighbor_lookup` now stores *only 344K* neighbor lists instead of 6.46M.  
- Computation is year-based: each year processes 344K rows vs. looping across all 6.46M repeatedly.  
- Vectorized `vapply` improves speed and avoids large allocations in nested loops.  
- Preserves original estimand, column alignment, and pipeline compatibility with existing pre-trained Random Forest model.  

**Expected runtime**: Drops from 86+ hours to a few hours (or less) on a 16 GB laptop, as most overhead from redundant neighbor recomputation and nested row-level lookups is eliminated.