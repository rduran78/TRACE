 **Diagnosis:**  
The current pipeline recomputes neighbor lookup **per variable** and iterates over 6.46M rows repeatedly, causing massive redundant work. The `build_neighbor_lookup()` function is efficient (O(n)), but repeatedly calling `compute_neighbor_stats()` for 5 variables over millions of rows involves multiple passes through `neighbor_lookup`, which is costly in both time and memory. This explains the 86+ hours estimate.

---

**Optimization Strategy:**  
- **Precompute `neighbor_lookup` once** (already done in your outer loop).
- **Compute all neighbor statistics in a single pass** instead of looping over each variable separately.
- Use **matrix operations** and `vapply`/`data.table` for vectorized speed.
- Avoid repeated large `lapply` calls by processing multiple variables together.
- Keep the Random Forest model intact; only optimize feature engineering.

---

**Optimized R Code:**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Build neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Variables for which we compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor stats in one pass
compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  vals_matrix <- as.matrix(data[, ..vars])  # nrow x nvars
  n <- nrow(data)
  res <- matrix(NA_real_, n, length(vars) * 3)  # max, min, mean for each var
  colnames(res) <- as.vector(outer(vars, c("_nbr_max", "_nbr_min", "_nbr_mean"), paste0))
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    nbr_vals <- vals_matrix[idx, , drop = FALSE]
    nbr_vals <- nbr_vals[apply(!is.na(nbr_vals), 1, any), , drop = FALSE]  # filter rows with all NA
    if (nrow(nbr_vals) == 0) next
    res[i, seq(1, ncol(res), 3)] <- apply(nbr_vals, 2, max, na.rm = TRUE)
    res[i, seq(2, ncol(res), 3)] <- apply(nbr_vals, 2, min, na.rm = TRUE)
    res[i, seq(3, ncol(res), 3)] <- apply(nbr_vals, 2, mean, na.rm = TRUE)
  }
  res
}

# Apply the optimized function and bind columns
neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)
cell_data <- cbind(cell_data, neighbor_stats)

# Now you can proceed with prediction using the pre-trained Random Forest model
# Example:
# preds <- predict(trained_rf_model, newdata = cell_data)
```

---

**Why This Is Faster:**  
- Only **one pass** through `neighbor_lookup` instead of 5.
- Matrix-based operations reduce R interpreter overhead.
- Memory footprint is controlled by preallocating the result matrix.
- Avoids repeated joins or repeated filtering across variables.

This approach can reduce runtime from **86+ hours to a few hours** on a standard laptop, given the scale of 6.46M rows. For even better performance, consider **parallelizing the loop** with `future.apply` or `parallel` if resources allow.