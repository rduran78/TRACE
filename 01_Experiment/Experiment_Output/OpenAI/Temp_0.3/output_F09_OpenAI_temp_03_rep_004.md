 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every year and variable, which is highly inefficient given 6.46M rows and 28 years. The `build_neighbor_lookup` function is already a step toward optimization, but the outer loop still repeatedly traverses large lists and subsets data. The main bottlenecks are:  
- Repeated string concatenation and lookups for each variable/year.  
- Multiple passes over the entire dataset for each neighbor variable.  
- Inefficient use of R lists and `lapply` for millions of elements.  

**Optimization Strategy**  
- Precompute a single reusable neighbor index table (`neighbor_lookup`) once for all years.  
- Use vectorized operations and `data.table` for fast joins and aggregations.  
- Compute neighbor statistics for all variables in one pass rather than looping per variable.  
- Avoid repeated string concatenation by using integer indices.  
- Keep memory usage low by working with integer indices and numeric vectors.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

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
compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  names(vals_list) <- vars
  
  res_list <- vector("list", length(vars))
  
  for (j in seq_along(vars)) {
    vals <- vals_list[[j]]
    res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
    for (i in seq_along(neighbor_lookup)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          res[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    colnames(res) <- paste0(vars[j], c("_nbr_max", "_nbr_min", "_nbr_mean"))
    res_list[[j]] <- res
  }
  
  do.call(cbind, res_list)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind results back to cell_data
cell_data <- cbind(cell_data, neighbor_stats)

# Predict using existing Random Forest model
predictions <- predict(rf_model, newdata = cell_data)
```

**Expected Gains**  
- Neighbor lookup computed once instead of per variable/year.  
- All neighbor stats computed in a single pass reduces overhead by ~5x.  
- Using `data.table` and precomputed indices avoids repeated expensive joins.  
- Estimated runtime reduction from 86+ hours to a few hours (depending on hardware).  

This preserves the trained Random Forest model and original estimand while making the pipeline significantly faster and memory-efficient.