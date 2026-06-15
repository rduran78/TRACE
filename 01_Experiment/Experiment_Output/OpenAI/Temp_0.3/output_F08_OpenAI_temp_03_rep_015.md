 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each cell-year row by repeatedly scanning neighbor indices. This results in ~6.46M iterations, each performing lookups and aggregations, even though the neighbor structure is static across years. The repeated lapply calls and row-wise operations cause massive overhead and memory churn.

**Optimization Strategy**  
- **Exploit static neighbor relationships**: Precompute a neighbor index matrix for cells (not cell-years) once.
- **Vectorize by year**: For each year, slice the data for that year and compute neighbor stats in bulk using matrix operations.
- **Avoid repeated lapply per row**: Use `vapply` or direct matrix aggregation.
- **Memory efficiency**: Process year-by-year to keep memory footprint low.
- **Preserve estimand**: Ensure the computed stats match the original logic (exclude `NA`s, return `NA` if all neighbors are `NA`).

**Optimized R Code**

```r
# Precompute static neighbor index list for cells
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_ref[as.character(neighbor_ids)])
  })
}

# Compute neighbor stats for one variable and one year
compute_year_neighbor_stats <- function(var_vec, neighbor_lookup) {
  n <- length(var_vec)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    vals <- var_vec[idx]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0) next
    res[i, ] <- c(max(vals), min(vals), mean(vals))
  }
  res
}

# Main optimized pipeline
compute_all_neighbor_features <- function(data, id_order, neighbors, vars) {
  neighbor_lookup <- build_static_neighbor_lookup(id_order, neighbors)
  years <- sort(unique(data$year))
  
  # Prepare result storage
  out_list <- vector("list", length(vars))
  names(out_list) <- vars
  for (v in vars) {
    out_list[[v]] <- matrix(NA_real_, nrow = nrow(data), ncol = 3)
  }
  
  for (yr in years) {
    idx_year <- which(data$year == yr)
    # Map to cell index
    cell_idx <- match(data$id[idx_year], id_order)
    
    for (v in vars) {
      var_vec <- rep(NA_real_, length(id_order))
      var_vec[cell_idx] <- data[[v]][idx_year]
      stats_mat <- compute_year_neighbor_stats(var_vec, neighbor_lookup)
      out_list[[v]][idx_year, ] <- stats_mat[cell_idx, ]
    }
  }
  
  # Bind results back to data
  for (v in vars) {
    colnames(out_list[[v]]) <- paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    data <- cbind(data, out_list[[v]])
  }
  data
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

**Why it’s faster**  
- Neighbor lookup built once for 344k cells (not 6.46M rows).
- Year-by-year processing avoids huge intermediate objects.
- Inner loop runs over 344k cells instead of 6.46M rows.
- No repeated string concatenation or hash lookups per row.

**Expected performance**  
- Complexity reduced from O(N_rows × avg_neighbors) to O(N_cells × avg_neighbors × N_years).
- On a standard laptop, this should reduce runtime from 86+ hours to a few hours or less, depending on I/O and vectorization overhead.

**Preserves**  
- Original Random Forest model.
- Original numerical estimand (same logic for NA handling and stats).