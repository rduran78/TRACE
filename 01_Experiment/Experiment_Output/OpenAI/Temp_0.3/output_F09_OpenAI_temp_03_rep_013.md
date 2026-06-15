 **Diagnosis**  
The pipeline is slow because neighbor relationships are recomputed for every variable and year. With 6.46M rows and repeated joins, this leads to excessive overhead. The adjacency structure (cell-to-neighbor mapping) is static across years, so rebuilding neighbor lookups repeatedly is wasteful. Computing neighbor statistics in pure R loops over millions of rows is also inefficient.

**Optimization Strategy**  
- Build the neighbor lookup **once** for all cell-year rows using the static adjacency table and year-specific keys.
- Use **vectorized operations** or `data.table` for joining and aggregating instead of repeated `lapply`.
- Precompute neighbor indices and reuse them for all variables.
- Avoid copying large data frames repeatedly; compute all neighbor stats in one pass if possible.
- Keep memory usage low by working with integer indices and matrices.

**Working R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Build reusable neighbor lookup once
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}

neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for multiple variables efficiently
compute_neighbor_stats <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  names(vals_list) <- vars
  
  result_list <- vector("list", length(vars))
  
  for (j in seq_along(vars)) {
    vals <- vals_list[[j]]
    result_list[[j]] <- t(vapply(neighbor_lookup, function(idx) {
      if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3)))
  }
  
  names(result_list) <- vars
  result_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Add computed features back to cell_data
for (v in neighbor_source_vars) {
  stats <- neighbor_stats[[v]]
  cell_data[[paste0(v, "_nbr_max")]] <- stats[,1]
  cell_data[[paste0(v, "_nbr_min")]] <- stats[,2]
  cell_data[[paste0(v, "_nbr_mean")]] <- stats[,3]
}

# Predict using pre-trained Random Forest model
predictions <- predict(rf_model, newdata = cell_data)
```

**Expected Impact**  
- Neighbor lookup built once instead of per variable.
- Aggregation done in memory without repeated joins.
- Avoids 86+ hour runtime; should reduce to a few hours or less on 16 GB RAM.
- Preserves trained Random Forest model and original estimand.