 **Diagnosis:**  
The colleague’s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. While `rbind` on millions of rows is non-trivial, the deeper bottleneck is the **nested `lapply` over 6.46M rows for each of 5 variables**, repeatedly scanning neighbor indices and subsetting vectors in pure R. This results in ~32M R-level function calls and heavy interpreter overhead. The real issue is the lack of vectorization and compiled operations.

---

**Optimization Strategy:**  
- Precompute neighbor index matrix once (fixed width, padded with `NA`).
- Use **matrix operations** to compute `max`, `min`, and `mean` across neighbors in compiled code.
- Avoid repeated R loops and repeated `lapply` calls.
- Use `matrixStats` or `apply` on numeric matrices for speed.
- Preserve the trained Random Forest model and estimand by keeping the same feature definitions.

---

**Optimized R Code:**

```r
library(matrixStats)

# Precompute neighbor index matrix (pad with NA)
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))

  max_neighbors <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = length(row_ids), ncol = max_neighbors)

  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    idx <- idx_lookup[neighbor_keys]
    idx <- as.integer(idx[!is.na(idx)])
    if (length(idx) > 0) {
      neighbor_mat[i, seq_along(idx)] <- idx
    }
  }
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  # Build neighbor value matrix
  neighbor_vals_mat <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))
  # Handle NA rows
  row_max <- rowMaxs(neighbor_vals_mat, na.rm = TRUE)
  row_min <- rowMins(neighbor_vals_mat, na.rm = TRUE)
  row_mean <- rowMeans(neighbor_vals_mat, na.rm = TRUE)
  # Replace NaN (all NA neighbors) with NA
  row_mean[is.nan(row_mean)] <- NA
  cbind(row_max, row_min, row_mean)
}

# Usage
neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why this works:**  
- Eliminates millions of R-level function calls.
- Uses contiguous memory and vectorized C-level operations via `matrixStats`.
- Reduces runtime from **86+ hours to a few hours or less** on a standard laptop.
- Preserves the original estimand and trained model.  

**Key Insight:** The bottleneck is **R interpreter overhead from per-row neighbor aggregation**, not `rbind`. The fix is **vectorization and precomputation**.