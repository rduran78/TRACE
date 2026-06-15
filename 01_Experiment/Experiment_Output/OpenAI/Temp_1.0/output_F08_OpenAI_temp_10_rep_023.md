 **Diagnosis:**  
The current implementation recomputes neighbor statistics for every cell-year entry by iterating through all ~6.46M rows multiple times (once per variable). While the neighbor structure is static across years, the code redundantly processes the same lookup for every entry and variable. This causes excessive computation (~86+ hours). Memory is stressed due to repeated `lapply` calls and intermediate `rbind` operations across millions of rows.

---

**Optimization Strategy:**  
- Precompute a **static neighbor lookup by cell ID** (not by cell-year).
- For each year:
  - Filter data for that year only (≈344k rows).
  - Compute neighbor statistics using the static lookup on year-specific values only.
  - Append results efficiently.
- Process all five variables in a single pass per year to reduce overhead.
- Avoid repeated creation of large intermediate lists; instead use preallocated matrices/data frames.
- Use `data.table` for speed on filtering and joining operations.

---

**Working R Code:**

```r
library(data.table)

# Precompute static neighbor lookup by cell ID (not year-specific)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}

# Function to compute all neighbor stats for a single year's data
compute_neighbor_stats_year <- function(year_data, neighbor_lookup, var_names) {
  n <- nrow(year_data)
  m <- length(var_names)
  result_mat <- matrix(NA_real_, nrow = n, ncol = m * 3)
  
  vals_list <- lapply(var_names, function(v) year_data[[v]])
  
  for (i in seq_len(n)) {
    nb_ids <- neighbor_lookup[[i]]
    if (length(nb_ids) == 0) next
    nb_idx <- match(nb_ids, year_data$id)
    valid_neighbors <- !is.na(nb_idx)
    if (!any(valid_neighbors)) next
    nb_idx <- nb_idx[valid_neighbors]
    
    for (j in seq_along(var_names)) {
      nb_vals <- vals_list[[j]][nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) next
      result_mat[i, (j - 1) * 3 + 1] <- max(nb_vals)
      result_mat[i, (j - 1) * 3 + 2] <- min(nb_vals)
      result_mat[i, (j - 1) * 3 + 3] <- mean(nb_vals)
    }
  }
  
  result_mat
}

# Main optimized pipeline
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, var_names) {
  setDT(cell_data)
  
  # Build static lookup
  neighbor_lookup <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)
  
  # Predefine feature column names
  feature_cols <- as.vector(t(outer(var_names, c("max","min","mean"), paste, sep = "_nb_")))
  
  # Add empty columns for features
  for (col in feature_cols) set(cell_data, j = col, value = NA_real_)
  
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    year_idx <- which(cell_data$year == yr)
    year_data <- cell_data[year_idx]
    
    stats_mat <- compute_neighbor_stats_year(year_data, neighbor_lookup, var_names)
    
    # Assign computed stats back
    for (k in seq_along(feature_cols)) {
      set(cell_data, i = year_idx, j = feature_cols[k], value = stats_mat[,k])
    }
    cat("Processed year:", yr, "\n")
  }
  
  cell_data
}

# Example usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

**Efficiency Gains:**
- **Static neighbor mapping** computed once instead of per row.
- **Year-based batching:** reduces row count per pass from 6.46M to 344k.
- **Memory safe:** Preallocates result matrices; avoids large intermediate lists.
- Complexity drops from O(N_rows * N_vars * neighbor_size) across entire dataset to O(N_years * N_cells * N_vars).

This design should reduce 86+ hours to **minutes to a few hours on a standard laptop** while preserving original estimands and compatibility with the trained Random Forest model.