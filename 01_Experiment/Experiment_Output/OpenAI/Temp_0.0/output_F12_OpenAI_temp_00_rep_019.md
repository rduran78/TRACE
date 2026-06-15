 **Diagnosis:**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `rbind` on millions of rows is non-trivial, the deeper bottleneck lies in the **outer loop repeatedly scanning the entire 6.46M-row dataset for each variable**. Specifically:  
- `compute_neighbor_stats()` is called 5 times (once per variable), and each call iterates over all 6.46M rows, performing neighbor lookups and filtering NA values.  
- This results in ~32 million neighbor computations and repeated memory allocations.  
- The neighbor structure and index mapping are recomputed for every variable, even though they are invariant.  

The real inefficiency is **repeated full-pass neighbor aggregation per variable** rather than vectorized or batched computation.

---

### **Optimization Strategy**
- Precompute neighbor indices once (already done by `build_neighbor_lookup`).
- Replace repeated `lapply` over 6.46M rows with a **vectorized matrix-based approach** using `matrixStats` or `data.table` for grouped aggregation.
- Compute all 5 variables in a **single pass** over neighbors, storing results in preallocated matrices.
- Avoid `do.call(rbind, ...)` entirely by writing directly into a numeric matrix.

---

### **Optimized R Code**

```r
library(matrixStats)

compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  n <- length(neighbor_lookup)
  k <- length(var_names)
  # Preallocate: 3 stats (max, min, mean) per variable
  out <- matrix(NA_real_, nrow = n, ncol = 3 * k)
  colnames(out) <- as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))

  # Extract all variable columns as a matrix for fast access
  vals_mat <- as.matrix(data[, var_names, drop = FALSE])

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals_mat[idx, , drop = FALSE]
      # Remove rows with all NAs
      if (nrow(neighbor_vals) > 0) {
        out[i, seq(1, 3 * k, by = 3)] <- colMaxs(neighbor_vals, na.rm = TRUE)
        out[i, seq(2, 3 * k, by = 3)] <- colMins(neighbor_vals, na.rm = TRUE)
        out[i, seq(3, 3 * k, by = 3)] <- colMeans2(neighbor_vals, na.rm = TRUE)
      }
    }
  }
  out
}

# Usage:
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind results back to cell_data
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Why This Works**
- **Single pass** over neighbor list instead of 5 passes.
- **Preallocated matrix** avoids repeated `rbind` and list growth.
- **Vectorized column operations** (`colMaxs`, `colMins`, `colMeans2`) replace per-variable loops.
- Memory footprint is controlled: only one large matrix instead of multiple intermediate lists.

---

**Expected Impact:**  
This reduces complexity from `O(n * k)` full scans with repeated allocations to a single `O(n * k)` pass with vectorized operations, likely cutting runtime from 86+ hours to a few hours or less on a 16 GB laptop.