 **Diagnosis:**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `rbind` on millions of rows is non-trivial, the deeper bottleneck lies in the **outer loop repeatedly scanning the entire dataset for each variable**. Specifically:

- `compute_neighbor_stats()` is called **5 times** (once per variable) over **6.46 million rows**, performing repeated neighbor lookups and filtering.
- Each call iterates over all rows and recomputes neighbor statistics from scratch, even though the neighbor structure is fixed.
- This results in ~32 million neighbor computations and redundant memory allocations.

The real inefficiency is the **repeated full pass over the dataset and redundant neighbor aggregation**, not the final `rbind`.

---

### **Optimization Strategy**
- Precompute neighbor indices once (already done by `build_neighbor_lookup`).
- Vectorize neighbor aggregation across **all variables in one pass**, avoiding repeated loops.
- Use a **matrix** for results instead of repeatedly binding lists.
- Leverage `vapply` or preallocated arrays for speed and memory efficiency.

---

### **Optimized Implementation**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  n <- length(neighbor_lookup)
  k <- length(var_names)
  # Preallocate result matrix: 3 stats per variable
  result <- matrix(NA_real_, nrow = n, ncol = 3 * k)
  colnames(result) <- as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))

  vals_list <- lapply(var_names, function(v) data[[v]])

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_len(k)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      result[i, (3 * (j - 1) + 1):(3 * j)] <- c(
        max(neighbor_vals),
        min(neighbor_vals),
        mean(neighbor_vals)
      )
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

---

### **Why This Works**
- **Single pass** over 6.46M rows instead of 5 passes.
- Eliminates repeated `do.call(rbind, ...)` and list allocations.
- Uses preallocated matrix for results → minimal memory churn.
- Preserves original estimand and trained Random Forest model.

---

**Expected Impact:**  
This reduces complexity from `O(n * k)` full recomputations with heavy overhead to a single `O(n * k)` pass with minimal allocations. On a 16 GB laptop, this should cut runtime from **86+ hours to a few hours** (or less with further parallelization).