 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for all 6.46 million rows by iterating through each row and looking up neighbors dynamically. This results in repeated work across years, even though the neighbor structure is static. The complexity is roughly `O(N * avg_neighbors)` for each variable, multiplied by 28 years and 5 variables, leading to massive overhead. Additionally, the `lapply` approach with repeated indexing and string concatenation is inefficient in R for such large data.

---

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute a neighbor index for each cell (not cell-year) once.
- **Process by year in blocks**: For each year, extract the relevant slice of data and compute neighbor stats using vectorized operations.
- **Avoid repeated string operations**: Use integer indices instead of concatenated keys.
- **Use matrix operations**: Compute max, min, and mean for neighbors using `vapply` or `apply` on precomputed neighbor lists.
- **Memory efficiency**: Work year-by-year to avoid loading all intermediate results in memory.

---

**Optimized R Code**

```r
# Precompute neighbor lookup at cell level (static)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is a list of integer vectors (spdep::nb)
  # Return as-is but ensure it's aligned with id_order
  neighbors
}

compute_neighbor_stats_year <- function(values, neighbor_lookup) {
  # values: numeric vector for one year, length = number of cells
  n <- length(values)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- values[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Main optimized loop
add_neighbor_features <- function(cell_data, id_order, neighbor_lookup, vars) {
  # Assume cell_data has columns: id, year, and vars
  years <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  
  # Preallocate result list
  all_results <- vector("list", length(vars))
  names(all_results) <- vars
  
  for (var_name in vars) {
    all_results[[var_name]] <- matrix(NA_real_, nrow = nrow(cell_data), ncol = 3)
  }
  
  for (yr in years) {
    year_idx <- which(cell_data$year == yr)
    # Ensure order matches id_order
    year_data <- cell_data[year_idx, ]
    year_data <- year_data[match(id_order, year_data$id), ]
    
    for (var_name in vars) {
      stats_mat <- compute_neighbor_stats_year(year_data[[var_name]], neighbor_lookup)
      # Place back into all_results
      all_results[[var_name]][year_idx, ] <- stats_mat
    }
  }
  
  # Bind results to cell_data
  for (var_name in vars) {
    colnames(all_results[[var_name]]) <- paste0(var_name, "_nbr_", c("max", "min", "mean"))
    cell_data <- cbind(cell_data, all_results[[var_name]])
  }
  
  cell_data
}

# Usage
neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features(cell_data, id_order, neighbor_lookup, neighbor_source_vars)
```

---

**Why This Works**  
- **Static neighbor structure**: We compute neighbors once at the cell level.
- **Year-by-year processing**: Reduces memory footprint and avoids redundant computation.
- **Vectorized assignment**: Avoids repeated string concatenation and dynamic lookups.
- **Preserves estimand**: The logic for max, min, and mean remains identical to the original.

**Expected Performance**  
This approach reduces complexity from `O(N * years * vars)` with heavy overhead to `O(cells * avg_neighbors * years * vars)` with minimal overhead and efficient memory use. On a standard laptop, this should reduce runtime from 86+ hours to a few hours or less, depending on I/O speed.