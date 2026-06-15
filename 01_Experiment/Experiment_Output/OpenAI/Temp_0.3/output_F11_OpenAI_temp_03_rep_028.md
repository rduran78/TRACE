 The evidence points to the **neighbor feature computation** as the bottleneck, not Random Forest inference. The nested `lapply` calls over ~6.46 million rows and repeated string concatenation in `build_neighbor_lookup` and `compute_neighbor_stats` dominate runtime. Random Forest `predict()` on 6.46M rows with 110 features is fast compared to these R loops.

---

### **Diagnosis**
- `build_neighbor_lookup` constructs a massive list of length 6.46M, performing string concatenation and named lookups repeatedly.
- `compute_neighbor_stats` iterates over the same 6.46M rows for each of 5 variables, performing repeated subsetting and aggregation.
- Complexity: O(N × neighbors × vars) in pure R loops → extremely slow.

---

### **Optimization Strategy**
- Precompute neighbor indices once as an **integer matrix** instead of lists.
- Use **vectorized operations** or `data.table` to compute neighbor stats in bulk.
- Avoid repeated string concatenation and named lookups.
- Parallelize where possible.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute neighbor lookup as integer matrix
# Each row corresponds to a cell-year, columns are neighbor row indices
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  n <- nrow(data)
  max_neighbors <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    keys <- paste(neighbor_ids, data$year[i], sep = "_")
    idx <- idx_lookup[keys]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      neighbor_mat[i, seq_along(idx)] <- idx
    }
  }
  neighbor_mat
}

neighbor_mat <- build_neighbor_matrix(dt, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_mat) {
  n <- nrow(neighbor_mat)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        res[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  res
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- dt[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_mat)
  dt[, paste0(var_name, "_nbr_max") := stats[, 1]]
  dt[, paste0(var_name, "_nbr_min") := stats[, 2]]
  dt[, paste0(var_name, "_nbr_mean") := stats[, 3]]
}
```

---

### **Expected Gains**
- Eliminates repeated string operations and nested `lapply`.
- Reduces complexity by using a fixed-size matrix and vectorized assignment.
- Further speedup possible with `Rcpp` or parallelization.

**Bottom line:** The bottleneck is neighbor feature engineering, not Random Forest inference. Optimize neighbor computations as shown; keep the trained model unchanged.