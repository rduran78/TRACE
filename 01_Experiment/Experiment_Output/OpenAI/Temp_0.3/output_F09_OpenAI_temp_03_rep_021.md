 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each variable across all 6.46 million rows, which is highly inefficient. The bottleneck arises from redundant lookups and repeated iteration over large lists. Building a reusable neighbor lookup table is a good strategy because spatial adjacency does not change over time. The main issues:  
- `build_neighbor_lookup` is called once, but `compute_neighbor_stats` iterates over all rows for each variable, causing 5 full passes over 6.46M rows.  
- Each pass involves list operations and repeated NA filtering, which is slow in R.  
- Memory pressure from large lists and repeated `do.call(rbind, ...)` calls.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** for all cell-year rows using the reusable lookup table.  
2. **Vectorize neighbor statistics computation** by reducing repeated list traversals and using matrix operations where possible.  
3. **Process variables in a single pass** by computing max, min, and mean for all neighbor source variables simultaneously.  
4. Use `data.table` for fast joins and column operations.  
5. Avoid copying large data frames repeatedly; append results efficiently.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table for efficiency
cell_data <- as.data.table(cell_data)

# Precompute neighbor lookup once
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

# Compute neighbor stats for all variables in one pass
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  vals_list <- lapply(var_names, function(v) data[[v]])
  names(vals_list) <- var_names
  
  result_list <- vector("list", length(neighbor_lookup))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      result_list[[i]] <- rep(NA_real_, length(var_names) * 3)
    } else {
      stats <- unlist(lapply(vals_list, function(vals) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
        c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }))
      result_list[[i]] <- stats
    }
  }
  
  do.call(rbind, result_list)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_matrix <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Assign column names
col_names <- unlist(lapply(neighbor_source_vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
stats_dt <- as.data.table(stats_matrix)
setnames(stats_dt, col_names)

# Bind to original data
cell_data <- cbind(cell_data, stats_dt)

# Random Forest prediction (model preserved)
predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Why This Is Faster**  
- Neighbor lookup is computed once and reused.  
- All neighbor statistics for 5 variables are computed in a single loop instead of 5 separate loops.  
- Avoids repeated `rbind` calls inside loops by building a final matrix.  
- Uses `data.table` for efficient column binding and memory handling.  

This approach should reduce runtime from 86+ hours to a few hours (depending on I/O and CPU), making it feasible on a 16 GB RAM laptop without retraining the Random Forest model or changing the estimand.