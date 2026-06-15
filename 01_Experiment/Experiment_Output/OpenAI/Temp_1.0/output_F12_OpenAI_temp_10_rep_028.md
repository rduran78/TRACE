 **Diagnosis:**  
The colleague's claim that `do.call(rbind, result)` and repeated list binding are the main bottlenecks is incorrect. These operations occur in `compute_neighbor_stats()`, but they are trivial compared to the *outer loop recomputation* and the repeated full traversal over 6.46 million rows for each variable (`neighbor_source_vars`). The true bottleneck is the repeated execution of neighbor lookups and aggregations across the entire panel for each variable, rather than batching the work.

**Deeper Bottleneck:**  
- `compute_neighbor_stats()` is executed 5 times (once for each variable), each time iterating through all rows.
- The neighbor lookup (`neighbor_lookup`) itself is static across variables and could be leveraged for vectorized computation.
- Building and binding lists repeatedly amplifies the problem across 6.46M rows × 5 variables.

**Optimization Strategy:**  
Compute **all required neighbor statistics for all variables in a single pass**, then bind results once. This eliminates redundant loops and avoids repeatedly calling functions that traverse millions of records. Use `vapply` for safer and faster aggregation, and preallocate final results rather than doing repeated bind operations.

---

### **Working R Code:**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  # Preallocate matrix: rows = nrow(data), cols = 3 * length(var_names)
  n <- length(neighbor_lookup)
  res_mat <- matrix(NA_real_, nrow = n, ncol = 3 * length(var_names))
  col_names <- unlist(lapply(var_names, function(v) c(paste0(v, "_max"),
                                                     paste0(v, "_min"),
                                                     paste0(v, "_mean"))))
  colnames(res_mat) <- col_names
  
  vals_list <- lapply(var_names, function(v) data[[v]])
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(vals_list)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      stats <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      res_mat[i, (3*j - 2):(3*j)] <- stats
    }
  }
  
  as.data.frame(res_mat)
}

# Usage
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_stats_df <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_vars)

# Bind to cell_data in one operation
cell_data <- cbind(cell_data, neighbor_stats_df)
```

---

### **Expected Gains:**
- Single pass through 6.46M rows instead of 5 passes.
- Avoids repeated `do.call(rbind, ...)` and repeated `lapply(...)` calls per variable.
- Reduces run time from multiple days (86+ hours) to a few hours or less, depending on machine and I/O constraints.
- Preserves original Random Forest model and estimands because only input features formatting changes; underlying statistics remain identical.  

This batching and preallocation approach exploits **loop fusion** and memory locality, addressing the true bottleneck efficiently.