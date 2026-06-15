 The colleague’s diagnosis is incorrect. The main bottleneck is **not** `do.call(rbind, result)` or repeated list binding inside `compute_neighbor_stats()`. The real bottleneck is the **outer loop repeatedly scanning the entire 6.46M-row dataset for each variable**, performing redundant neighbor lookups and value filtering five times. This results in massive repeated work and memory churn.

### **Diagnosis**
- `compute_neighbor_stats()` is called **5 times**, each time iterating over all 6.46M rows and recomputing neighbor statistics from scratch.
- `neighbor_lookup` is static across variables, but the function repeatedly extracts and filters neighbor values for each variable.
- `do.call(rbind, result)` is linear in the number of rows and negligible compared to the repeated neighbor computations.
- The true cost driver: **5 × 6.46M neighbor scans** → ~32M neighbor operations.

### **Optimization Strategy**
- Compute **all neighbor stats for all 5 variables in a single pass** over `neighbor_lookup`.
- Avoid repeated filtering and repeated list traversals.
- Use **vectorized matrix operations** instead of repeated `lapply` calls.
- Preallocate a numeric matrix for results to eliminate repeated binding.

### **Optimized Implementation**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  n <- length(neighbor_lookup)
  k <- length(var_names)
  result <- matrix(NA_real_, nrow = n, ncol = 3 * k)
  colnames(result) <- as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))

  vals_list <- lapply(var_names, function(v) data[[v]])

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(var_names)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      base <- (j - 1) * 3
      result[i, base + 1L] <- max(neighbor_vals)
      result[i, base + 2L] <- min(neighbor_vals)
      result[i, base + 3L] <- mean(neighbor_vals)
    }
  }
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind back to cell_data
cell_data <- cbind(cell_data, neighbor_stats)
```

### **Why This Works**
- Single traversal of `neighbor_lookup` instead of 5.
- Eliminates repeated `do.call(rbind, ...)`.
- Preallocates result matrix → no dynamic growth.
- Reduces complexity from **O(5 × N × avg_neighbors)** to **O(N × avg_neighbors)**.

**Expected speedup:** From 86+ hours to a few hours (or less) on a 16 GB laptop, as redundant work is removed and memory allocations are minimized.