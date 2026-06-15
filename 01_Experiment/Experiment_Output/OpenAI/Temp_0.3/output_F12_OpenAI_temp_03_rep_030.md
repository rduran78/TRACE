 The colleague’s diagnosis is incorrect: the main bottleneck is **not** `do.call(rbind, result)` or repeated list binding in `compute_neighbor_stats()`. The real bottleneck is the **outer loop repeatedly scanning the entire 6.46M-row dataset five times**, each time performing millions of random index lookups and filtering. This is highly inefficient because:

- `compute_neighbor_stats()` is called once per variable (5 times), iterating over all rows (6.46M) each time.
- Neighbor lookups and NA filtering are recomputed for every variable, even though neighbors don’t change across variables.
- `do.call(rbind, result)` is negligible compared to the repeated full passes and millions of small allocations.

### **Optimization Strategy**
- Compute **all neighbor statistics for all variables in one pass** over `neighbor_lookup`.
- Avoid repeated indexing and filtering by pre-extracting all relevant columns as a matrix.
- Use **vectorized operations** and preallocation instead of building lists and binding repeatedly.

### **Optimized Approach**
- Convert `neighbor_source_vars` into a numeric matrix for fast column access.
- For each row, compute stats for all variables in one loop over neighbors.
- Return a single matrix of size `nrow(data) × (length(vars) × 3)` (max, min, mean per variable).

---

### **Optimized R Code**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  vals_mat <- as.matrix(data[vars])
  n <- nrow(vals_mat)
  k <- length(vars)
  
  # Preallocate result matrix: 3 stats per variable
  res <- matrix(NA_real_, nrow = n, ncol = k * 3)
  colnames(res) <- as.vector(t(outer(vars, c("max", "min", "mean"), paste, sep = "_")))
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals_mat[idx, , drop = FALSE]
    # Remove rows where all are NA
    if (all(is.na(neighbor_vals))) next
    res[i, seq(1, k * 3, by = 3)] <- apply(neighbor_vals, 2, max, na.rm = TRUE)
    res[i, seq(2, k * 3, by = 3)] <- apply(neighbor_vals, 2, min, na.rm = TRUE)
    res[i, seq(3, k * 3, by = 3)] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
  }
  
  res
}

# Usage:
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind back to original data frame
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Why This Fix Works**
- **Single pass** over 6.46M rows instead of 5 passes.
- Eliminates repeated `vals[idx]` extraction for each variable.
- Avoids repeated `do.call(rbind, ...)` and list allocations.
- Memory-efficient: preallocated matrix instead of millions of small vectors.

**Expected speedup:** Orders of magnitude (from 86+ hours to a few hours or less), within 16 GB RAM, preserving the original estimand and trained model.