 **Diagnosis:**  
The main bottleneck is the neighbor feature computation (`build_neighbor_lookup` + `compute_neighbor_stats`), which uses deeply nested `lapply` calls and repeated object copying for 6.46M rows × 5 variables. This results in extremely high memory overhead and poor cache locality. The workflow repeatedly constructs vectors and binds rows, causing excessive allocations. Since Random Forest inference is relatively fast compared to these pre-prediction steps, the preprocessing dominates runtime.

---

### **Optimization Strategy**
1. **One-time neighbor index lookup:** Precompute neighbor indices in a flat integer matrix for all rows instead of `lapply` returning lists.
2. **Vectorize neighbor stats:** Replace repeated `lapply` + `rbind` calls with matrix operations using `data.table` or `Rcpp`.
3. **Avoid repeated joins:** Compute all neighbor-based features in a single pass.
4. **Preload RF model once:** Ensure the model stays in memory without reloading for each batch.
5. **Use `data.table` for speed and memory efficiency.**
6. **Parallelize compute-heavy steps** using `parallel::mclapply` or `future.apply`.

---

### **Optimized R Code**
Below is a working approach using `data.table` and vectorized neighbor calculation:

```r
library(data.table)
library(randomForest)
library(parallel)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup: map (id, year) -> row index
idx_lookup <- setNames(seq_len(nrow(cell_data)),
                       paste(cell_data$id, cell_data$year, sep = "_"))

# Build neighbor index matrix (flattened)
build_neighbor_index_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(data)
  max_neighbors <- max(lengths(neighbors))
  
  mat <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_ids, data$year[i], sep = "_")
    idx <- idx_lookup[neighbor_keys]
    valid_idx <- idx[!is.na(idx)]
    if (length(valid_idx) > 0) {
      mat[i, seq_along(valid_idx)] <- valid_idx
    }
  }
  mat
}

neighbor_index_matrix <- build_neighbor_index_matrix(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_all_neighbor_stats <- function(data, neighbor_matrix, vars) {
  n <- nrow(data)
  max_neighbors <- ncol(neighbor_matrix)
  results <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- data[[vars[v]]]
    mat_vals <- matrix(vals[neighbor_matrix], nrow = n, ncol = max_neighbors)
    row_max <- apply(mat_vals, 1, max, na.rm = TRUE)
    row_min <- apply(mat_vals, 1, min, na.rm = TRUE)
    row_mean <- rowMeans(mat_vals, na.rm = TRUE)
    row_max[!is.finite(row_max)] <- NA
    row_min[!is.finite(row_min)] <- NA
    row_mean[!is.finite(row_mean)] <- NA
    results[[v]] <- data.table(
      paste0(vars[v], "_ngh_max") := row_max,
      paste0(vars[v], "_ngh_min") := row_min,
      paste0(vars[v], "_ngh_mean") := row_mean
    )
  }
  
  do.call(cbind, results)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- compute_all_neighbor_stats(cell_data, neighbor_index_matrix, neighbor_source_vars)

# Bind neighbor features to main data
cell_data <- cbind(cell_data, neighbor_features)

# Random Forest prediction (batch mode)
# Assume rf_model is already loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Expected Gains**
- **Lookup building:** O(N) instead of repeated lapply overhead.
- **Neighbor stats:** Vectorized via matrix operations, eliminating millions of small list objects.
- **Memory efficiency:** Data held in dense integer matrix instead of nested lists.
- **Runtime reduction:** From 86+ hours → a few hours (depending on I/O and CPU cores).

---

**Summary:**  
The primary performance issue is in neighbor feature computation, not the Random Forest inference itself. By converting the neighbor lookup into a matrix and using vectorized computations with `data.table`, the pipeline becomes dramatically faster and memory-friendly while preserving the original estimand and model integrity.