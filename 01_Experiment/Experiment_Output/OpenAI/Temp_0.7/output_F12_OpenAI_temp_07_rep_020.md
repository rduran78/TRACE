 **Diagnosis**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding in `compute_neighbor_stats()` are the main bottlenecks is **incorrect**. While `do.call(rbind, ...)` on a 6.46M-row list is costly, the deeper bottleneck lies in the **outer loop repeatedly scanning the entire neighbor list (length ≈ 6.46M), recomputing neighbor statistics 5 times**—once for each variable. This multiplies the cost by 5. Each pass performs millions of lookups and filtering operations (`!is.na`), which dominate runtime.

**Correct Optimization Strategy**  
- Avoid recomputing neighbor stats for each variable.
- Compute all required neighbor summaries in a **single pass** over `neighbor_lookup`.
- Store results in a preallocated `matrix` or `data.table` to eliminate repeated `rbind` and redundant neighbor traversals.
- Keep the trained Random Forest model and numerical estimands unchanged.

---

### **Optimized Approach**
- Use `vapply` or vectorized loops for speed.
- Preallocate a numeric matrix of shape `nrow(data) × (length(neighbor_source_vars) * 3)`.
- Compute max, min, mean for all variables in one traversal.

---

### **Working R Code**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  n <- length(neighbor_lookup)
  k <- length(var_names)
  out <- matrix(NA_real_, nrow = n, ncol = k * 3)
  colnames(out) <- as.vector(t(outer(var_names, c("max", "min", "mean"), paste, sep = "_")))

  # Pre-extract columns to avoid repeated $ lookups
  vals_list <- lapply(var_names, function(v) data[[v]])

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(vals_list)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      base <- (j - 1) * 3
      out[i, base + 1] <- max(neighbor_vals)
      out[i, base + 2] <- min(neighbor_vals)
      out[i, base + 3] <- mean(neighbor_vals)
    }
  }
  out
}

# Build neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all stats in one pass
neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind back to the main data.frame
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Expected Gains**
- Reduces 5 full passes to **1 pass** over 6.46M rows.
- Eliminates repeated large `rbind` calls.
- Memory efficient (matrix preallocation).
- Likely reduces runtime from 86+ hours to **a few hours** on a standard laptop.

**Preserves trained model, estimands, and pipeline semantics.**