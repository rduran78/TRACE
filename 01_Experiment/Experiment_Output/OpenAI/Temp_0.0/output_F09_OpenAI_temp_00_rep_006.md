 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, the neighbor lookup is recomputed repeatedly for each variable. This results in redundant work and excessive memory allocation. The main bottlenecks are:  
- Rebuilding neighbor relationships per row and per variable.  
- Repeated `lapply` calls over millions of rows.  
- Lack of vectorization and precomputation.  

**Optimization Strategy**  
- Build the neighbor lookup **once** for all cell-year rows.  
- Reuse this lookup for all variables.  
- Use **matrix operations** or `data.table` for speed.  
- Avoid repeated string concatenation inside loops.  
- Compute neighbor stats in a single pass per variable using precomputed indices.  

**Optimized R Code**  

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

# Compute neighbor stats efficiently
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
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
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  res
}

# Apply for all variables and bind results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_stats <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats(cell_data, neighbor_lookup, v))
all_stats_mat <- do.call(cbind, all_stats)

# Bind to original data
cell_data <- cbind(cell_data, all_stats_mat)
```

**Why This Is Faster**  
- Neighbor lookup is computed **once** and reused.  
- Each variable is processed in a single pass using precomputed indices.  
- Avoids repeated string operations and redundant loops.  
- Uses preallocated matrices for results.  

**Expected Performance Gain**  
This approach should reduce runtime from 86+ hours to a few hours (or less) on a 16 GB laptop, as the most expensive operations are now O(N) rather than O(N × V) with repeated overhead.  

**Preserves**  
- Original Random Forest model (no retraining).  
- Original estimand and feature definitions.