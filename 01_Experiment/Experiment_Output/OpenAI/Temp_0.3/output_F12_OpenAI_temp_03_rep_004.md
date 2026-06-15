 **Diagnosis:**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `rbind` on millions of rows is non-trivial, the deeper performance issue lies in the **outer loop repeatedly scanning the entire dataset for each variable**. For 6.46M rows × 5 variables, the pipeline recomputes neighbor statistics five times, each time iterating over all rows and performing list indexing and filtering. This results in **O(N × V × avg_neighbors)** complexity, where N ≈ 6.46M and V = 5. The repeated `lapply` over 6.46M elements dominates runtime, not the final `rbind`.

---

### **Correct Optimization Strategy**
- **Compute all neighbor stats in one pass** instead of looping over variables.
- Use **vectorized operations** and **preallocate a matrix** for results.
- Avoid repeated filtering and indexing by caching neighbor indices and values.
- Optionally, use `data.table` for fast column access and assignment.

---

### **Optimized Approach**
- Precompute `neighbor_lookup` once (already done).
- Extract all required columns (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`) into a matrix.
- For each row, compute max, min, mean for all variables in one loop, store in a preallocated numeric matrix.
- Bind the result back to `cell_data`.

---

### **Working R Code**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  # Extract relevant columns as matrix for fast access
  vals_mat <- as.matrix(data[, var_names, drop = FALSE])
  n <- length(neighbor_lookup)
  p <- length(var_names)
  
  # Preallocate result matrix: 3 stats per variable
  result <- matrix(NA_real_, nrow = n, ncol = p * 3)
  colnames(result) <- as.vector(t(outer(var_names, c("max", "min", "mean"), paste, sep = "_")))
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals_mat[idx, , drop = FALSE]
      # Remove rows with all NAs
      if (!all(is.na(neighbor_vals))) {
        result[i, seq(1, p*3, 3)] <- apply(neighbor_vals, 2, max, na.rm = TRUE)
        result[i, seq(2, p*3, 3)] <- apply(neighbor_vals, 2, min, na.rm = TRUE)
        result[i, seq(3, p*3, 3)] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
      }
    }
  }
  
  result
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind back to original data
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Expected Performance Gain**
- Reduces 5 full passes over 6.46M rows to **one pass**.
- Eliminates repeated `do.call(rbind, ...)`.
- Memory-efficient: preallocated matrix avoids repeated list growth.
- On a standard laptop, this should cut runtime from **86+ hours to a few hours** (or less with further parallelization).

**Key Insight:** The bottleneck is the repeated full-dataset neighbor computation per variable, not `rbind`. The fix is **one-pass multi-variable aggregation with preallocation**.