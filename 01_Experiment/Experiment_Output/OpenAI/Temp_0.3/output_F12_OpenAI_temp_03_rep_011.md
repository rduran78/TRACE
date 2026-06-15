 **Diagnosis:**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `rbind` on millions of rows is non-trivial, the dominant cost lies in the **outer loop repeatedly scanning 6.46M rows for each of 5 variables**. Each call to `compute_neighbor_stats()` iterates over every cell-year row and performs neighbor lookups and filtering, which is extremely expensive. The repeated full pass over the dataset (5×6.46M iterations) dwarfs the cost of `rbind`.

**Correct bottleneck:**  
The pipeline is recomputing neighbor statistics for each variable separately, even though neighbor indices and filtering logic are identical across variables. This causes **5× redundant neighbor traversal**. The real optimization is to compute all required neighbor summaries in a **single pass**.

---

### **Optimization Strategy**
- Precompute neighbor indices once (already done).
- In a single traversal of `neighbor_lookup`, compute max, min, and mean for **all 5 variables simultaneously**.
- Return a matrix or data frame with 15 columns (3 stats × 5 variables).
- Bind results once to `cell_data`.

This reduces complexity from `O(n × vars)` to `O(n)` for neighbor traversal.

---

### **Optimized R Code**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  # Extract all relevant columns as a matrix for fast access
  vals_mat <- as.matrix(data[var_names])
  n <- length(neighbor_lookup)
  p <- length(var_names)
  
  # Preallocate result matrix: rows = n, cols = p * 3 (max, min, mean)
  result <- matrix(NA_real_, nrow = n, ncol = p * 3)
  colnames(result) <- as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals_mat[idx, , drop = FALSE]
    # Remove rows with all NAs
    if (all(is.na(neighbor_vals))) next
    
    # Compute stats column-wise
    result[i, seq(1, p * 3, by = 3)] <- apply(neighbor_vals, 2, max, na.rm = TRUE)
    result[i, seq(2, p * 3, by = 3)] <- apply(neighbor_vals, 2, min, na.rm = TRUE)
    result[i, seq(3, p * 3, by = 3)] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
  }
  
  as.data.frame(result)
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind to original data
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Why This Works**
- Eliminates 4 redundant full passes over 6.46M rows.
- Reduces `lapply` calls from 5×6.46M to 6.46M.
- Avoids repeated `do.call(rbind, ...)`.
- Memory footprint stays manageable: result matrix is ~6.46M × 15 doubles (~774 MB), feasible on 16 GB RAM.

**Expected speedup:**  
From 86+ hours to a few hours (dominated by a single neighbor traversal and matrix ops).

**Preserves:**  
- Original estimand.
- Trained Random Forest model.
- Numerical correctness.