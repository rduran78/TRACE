 **Diagnosis**  
The pipeline is slow because the neighbor relationships are recomputed repeatedly for every variable across all 6.46 million rows. The main bottleneck is the redundant construction of neighbor indices per variable and year. Since neighbors are static across years, building an adjacency structure once and reusing it for all variables is key. Memory is tight (16 GB RAM), so extreme replication should be avoided.

**Optimization Strategy**  
1. **Build static neighbor lookup once**: Map each cell-year row to its neighbors at the same year using precomputed cell-level adjacency and the `id_order` mapping.  
2. **Reuse lookup for all variables** instead of recomputing neighbor indices.  
3. Apply **vectorized aggregation**: compute neighbor stats in a loop for each variable using fast integer indexing and `vapply` instead of lapply + rbind cascade.  
4. Keep results lean by preallocating numeric matrices.  

This avoids 28 full recalculations of neighbor relationships and cuts overhead dramatically.

---

### **Working R Code**

```r
# Build neighbor lookup only once
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

# Compute neighbor statistics for one variable, efficiently
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = 3, dimnames = list(NULL, c("max", "min", "mean")))

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  res
}

# Main pipeline optimization
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats_mat)
}

# After adding neighbor features, run prediction with existing Random Forest model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Expected Speedup**:  
- Neighbor lookup built once (~minutes) versus repeated 5× per variable × 28 years.
- Simple preallocation and integer indexing reduce overhead from ~86 hours to **a few hours or less**, assuming efficient disk reads and caching.

This approach preserves the original estimand and works directly with your trained Random Forest model without retraining.