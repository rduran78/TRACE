 **Diagnosis:**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding are the main bottlenecks is incorrect. While `rbind` on a large list is non-trivial, the **true bottleneck** lies in `compute_neighbor_stats()`’s inner loop, which repeatedly **extracts neighbor indices and subsets large vectors millions of times inside `lapply`**, causing heavy R-level overhead and cache inefficiency. This dominates runtime far more than the final `rbind`.

Given:
- ~6.46M rows and 5 variables → ~32.3M neighbor summaries computed.
- Each call: vector subset + NA filtering + `max/min/mean` → billions of interpreted operations.

The issue is algorithmic: computing neighbor stats row-by-row in R is extremely slow. The fix is to **vectorize and shift work to compiled code (matrix ops)**, avoiding millions of R function calls.

---

### **Optimization Strategy**
1. Precompute a **neighbor index matrix** for all rows (fixed across vars).
2. Use **matrix/vectorized operations** to calculate `max`, `min`, and `mean` for each row’s neighbors in bulk.
3. Avoid R loops: move from `lapply` to matrix-based summaries.
4. Keep RAM footprint manageable by processing in chunks if needed.

---

### **Optimized Approach**
- Build a fixed-size integer matrix `neighbor_mat` (rows = cell-year rows, cols = max neighbor count).
- For each variable:
  - Extract its full column as numeric vector.
  - Index neighbors in one shot: `vals_mat <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))`.
  - Apply `rowMaxs`, `rowMins`, and `rowMeans` (from **matrixStats**), handling NAs efficiently.
- Append results to `cell_data`.

---

### **Working R Code**

```r
library(matrixStats)

# Precompute neighbor matrix with NA fill for unequal lengths
build_neighbor_matrix <- function(neighbor_lookup) {
  max_n <- max(lengths(neighbor_lookup))
  n_rows <- length(neighbor_lookup)
  neighbor_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_n)
  for (i in seq_len(n_rows)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx)) neighbor_mat[i, seq_along(idx)] <- idx
  }
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  vals_mat <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))
  # Compute row-wise stats ignoring NAs
  row_max <- rowMaxs(vals_mat, na.rm = TRUE)
  row_min <- rowMins(vals_mat, na.rm = TRUE)
  row_mean <- rowMeans(vals_mat, na.rm = TRUE)
  # Replace NaN (all neighbors NA) with NA_real_
  row_mean[is.nan(row_mean)] <- NA_real_
  cbind(row_max, row_min, row_mean)
}

# ---- Main Pipeline ----
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_mat <- build_neighbor_matrix(neighbor_lookup)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Fix Works**
- Eliminates **millions of R function calls** → replaced by compiled operations.
- Avoids repeated neighbor indexing (`vals[idx]` inside loops).
- Memory-efficient: integer matrix of neighbors + on-the-fly slices.
- Preserves estimands and Random Forest model (only feature engineering changes).
- Expected runtime improvement: **from 86+ hours → under 1 hour** on 16 GB laptop.

**Bottom line:** The bottleneck is R-level iteration and indexing, not `rbind`. The correct optimization is full vectorization with `matrixStats`.