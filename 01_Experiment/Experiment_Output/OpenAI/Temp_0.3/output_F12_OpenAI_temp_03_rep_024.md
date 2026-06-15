 The colleague’s claim is incorrect. The main bottleneck is **not** `do.call(rbind, result)` or repeated list binding in `compute_neighbor_stats()`. The real bottleneck is the **outer loop repeatedly scanning the entire 6.46M-row dataset for each variable**, performing expensive neighbor lookups and subsetting operations five times. Each pass recomputes neighbor statistics from scratch, leading to massive redundant work and memory thrashing.

### **Diagnosis**
- `do.call(rbind, result)` is O(n) and relatively cheap compared to 5 full dataset scans and repeated neighbor indexing.
- The heavy cost comes from:
  - Iterating over 6.46M rows × 5 variables.
  - Repeated extraction of neighbor values from `vals[idx]` for each variable.
- Neighbor relationships are **static**, so we can compute all required stats in a single pass and avoid redundant lookups.

---

### **Optimization Strategy**
- Precompute neighbor indices once (already done in `neighbor_lookup`).
- Compute **all 5 variables’ neighbor stats in one vectorized pass**.
- Use a preallocated numeric matrix instead of repeated list binding.
- Avoid repeated subsetting by iterating once over `neighbor_lookup`.

---

### **Optimized R Code**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  n <- length(neighbor_lookup)
  k <- length(var_names)
  result <- matrix(NA_real_, nrow = n, ncol = k * 3,
                   dimnames = list(NULL, as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))))
  
  vals_list <- lapply(var_names, function(v) data[[v]])
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(var_names)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      base_col <- (j - 1) * 3
      result[i, base_col + 1L] <- max(neighbor_vals)
      result[i, base_col + 2L] <- min(neighbor_vals)
      result[i, base_col + 3L] <- mean(neighbor_vals)
    }
  }
  as.data.frame(result)
}

# Usage:
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats_df <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind to original data
cell_data <- cbind(cell_data, neighbor_stats_df)
```

---

### **Expected Impact**
- Reduces 5 full passes over 6.46M rows to **one pass**.
- Eliminates repeated `do.call(rbind, ...)`.
- Memory-efficient: preallocated matrix avoids incremental growth.
- Should cut runtime from **86+ hours to a few hours** on a 16 GB laptop.

**Preserves the trained Random Forest model and numerical estimand.**