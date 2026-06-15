 **Diagnosis:**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `rbind` on millions of rows is non-trivial, the deeper bottleneck lies in the **outer loop repeatedly scanning the entire 6.46M-row dataset for each variable**. Each call to `compute_neighbor_stats()` iterates over all rows and recomputes neighbor statistics from scratch, causing redundant work and massive memory churn. The neighbor relationships and index lookups are static, so recomputing them per variable is wasteful.

---

**Correct Optimization Strategy:**  
- Precompute **all neighbor statistics for all variables in one pass** instead of looping over variables.
- Use **vectorized operations** and **matrix preallocation** to avoid repeated list-to-matrix conversions.
- Keep the trained Random Forest model intact and preserve the original estimand.

---

**Optimized R Code:**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  n <- length(neighbor_lookup)
  k <- length(var_names)
  # Preallocate result matrix: rows = observations, cols = 3 stats per variable
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3 * k)
  colnames(result_mat) <- as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))

  # Extract all variable columns as a list for fast access
  vals_list <- lapply(var_names, function(v) data[[v]])

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_len(k)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      base_col <- (j - 1) * 3
      result_mat[i, base_col + 1L] <- max(neighbor_vals)
      result_mat[i, base_col + 2L] <- min(neighbor_vals)
      result_mat[i, base_col + 3L] <- mean(neighbor_vals)
    }
  }
  result_mat
}

# Usage:
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind back to cell_data
cell_data <- cbind(cell_data, neighbor_stats)
```

---

**Why This Works:**  
- Eliminates 5 full passes over 6.46M rows → now only **one pass**.
- Avoids repeated `do.call(rbind, ...)` calls by preallocating a numeric matrix.
- Reduces memory fragmentation and GC overhead.
- Preserves the trained model and original estimand.

**Expected Impact:**  
This change should reduce runtime from **86+ hours to a few hours** (or less), depending on I/O and CPU speed, because the algorithm now scales linearly with rows × neighbors instead of rows × neighbors × variables × repeated overhead.