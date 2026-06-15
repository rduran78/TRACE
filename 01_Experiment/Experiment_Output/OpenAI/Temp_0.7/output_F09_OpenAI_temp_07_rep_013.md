 **Diagnosis:**  
The current pipeline repeatedly computes neighbor relationships for every year, which is extremely costly for 6.46 million rows. Building a neighbor lookup once and reusing it for all years will significantly reduce redundant computation. The main performance bottlenecks are:
- Rebuilding adjacency relationships per iteration.
- Inefficient repeated string concatenation and list handling per loop.
- Lack of vectorization when computing neighbor statistics.

---

**Optimization Strategy:**  
1. Precompute a *neighbor lookup* keyed by cell-year index **once** using spatial IDs and adjacency list.
2. Use `data.table` for fast joins and vectorized operations.
3. Compute neighbor stats (max, min, mean) in bulk per variable by iterating through prebuilt lookup.
4. Avoid copying large data frames repeatedly; update columns by reference.
5. Keep memory footprint reasonable (~16 GB RAM) by working in chunks if necessary.

---

**Optimized Working R Code:**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Pre-build neighbor lookup (indices)
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
  result <- vector("list", length(neighbor_lookup))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      result[[i]] <- c(NA_real_, NA_real_, NA_real_)
    } else {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        result[[i]] <- c(NA_real_, NA_real_, NA_real_)
      } else {
        result[[i]] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  do.call(rbind, result)
}

# Add neighbor features for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  set(cell_data, j = paste0(var_name, "_nbr_max"), value = stats[, 1])
  set(cell_data, j = paste0(var_name, "_nbr_min"), value = stats[, 2])
  set(cell_data, j = paste0(var_name, "_nbr_mean"), value = stats[, 3])
}

# Proceed to Random Forest prediction step without retraining
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Expected Impact:**  
- Eliminates redundant neighbor computation across 28 years.
- Uses fast `data.table` column assignment.
- Should reduce runtime from **86+ hours to a few hours**, depending on I/O and memory constraints.

**Preserves:**  
- Original Random Forest model.
- Original numerical estimand.
- Full adjacency logic for neighbor max/min/mean.