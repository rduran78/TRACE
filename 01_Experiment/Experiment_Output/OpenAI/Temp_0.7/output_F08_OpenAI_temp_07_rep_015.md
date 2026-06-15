 **Diagnosis**  
The current implementation repeatedly scans the entire 6.46M-row dataset for each variable and year combination using a large `neighbor_lookup` list of length equal to total rows. This is extremely costly because:  
- Neighbor relationships (1.37M directed edges) are *static across years*, but the code recomputes neighbor indices for every row-year pair.  
- For 28 years × 344k cells, operations scale with O(N × neighbors) repeatedly.  
- Memory overhead: building and storing a 6.46M-element list of integer vectors and repeatedly computing stats across it for each variable is inefficient.  

**Optimization strategy**  
Leverage the static nature of neighbors:  
- Build a neighbor index **once per cell** (length = 344,208).  
- For each year, slice the relevant variable vector and compute neighbor statistics by vectorized aggregation using this fixed neighbor structure.  
- Avoid `lapply` over all 6.46M rows; process by year in blocks.  
- Preallocate a result matrix and write in chunks to reduce memory thrash.  

This reduces complexity from ~6.46M × neighbor operations to (28 years × 344k cells) operations with pre-built neighbor lists only once.  

**Working R code**  

```r
# Build static neighbor lookup (cell-to-cell, no year)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_ref[as.character(neighbor_ids)]
  })
}

# Compute neighbor stats per year
compute_neighbor_stats_by_year <- function(data, static_neighbors, var_name, years, id_order) {
  n_cells <- length(id_order)
  n_years <- length(years)
  out <- matrix(NA_real_, nrow = n_cells * n_years, ncol = 3)
  
  colnames(out) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  
  for (y_idx in seq_along(years)) {
    yr <- years[y_idx]
    yr_idx <- which(data$year == yr)
    vals <- setNames(data[[var_name]][yr_idx], data$id[yr_idx])
    # Ensure order matches id_order
    vals_vec <- vals[as.character(id_order)]
    
    stats_mat <- t(sapply(static_neighbors, function(nei) {
      if (length(nei) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals_vec[nei]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }))
    
    row_range <- ((y_idx - 1) * n_cells + 1):(y_idx * n_cells)
    out[row_range, ] <- stats_mat
  }
  
  out
}

# Main driver
id_order <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))

# Build static neighbor lookup once
static_neighbors <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_neighbor_features <- list()

for (var_name in neighbor_source_vars) {
  message("Processing ", var_name)
  feature_mat <- compute_neighbor_stats_by_year(cell_data, static_neighbors, var_name, years, id_order)
  all_neighbor_features[[var_name]] <- feature_mat
}

# Combine all new features into a single data.frame
neighbor_features_df <- do.call(cbind, all_neighbor_features)

# Bind to original data
# Ensure cell_data is sorted by id_order then year
cell_data <- cell_data[order(match(cell_data$id, id_order), cell_data$year), ]
cell_data <- cbind(cell_data, neighbor_features_df)

# Now proceed with Random Forest predictions as before
```

**Why this works:**  
- Neighbor structure (1.37M edges) used once, not replicated for 6.46M rows.  
- Processes year-by-year, only 344k rows at a time.  
- Preallocation avoids repeated `rbind`.  
- Preserves estimands and Random Forest model (no retraining).  

**Expected performance gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, because repeated redundant lookups and large list traversals are eliminated.  
Memory footprint also drops significantly by avoiding 6.46M-element neighbor list.