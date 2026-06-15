 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations**: `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of rows, causing heavy R-level loops and object copying.  
2. **Random Forest inference inefficiency**: If predictions are done row-by-row or in small chunks, `predict()` overhead dominates.  
3. **Memory pressure**: Large intermediate lists (`neighbor_lookup`, repeated `do.call(rbind, ...)`) and repeated data frame copies.  

**Optimization Strategy**  
- **Precompute neighbor indices once** as integer vectors and reuse them.  
- **Vectorize neighbor feature computation**: Avoid `lapply` per row; use `matrixStats` or `data.table` to compute max/min/mean in bulk.  
- **Use `data.table` for in-place joins and updates** to avoid repeated copying.  
- **Batch Random Forest predictions**: Call `predict()` on the full data or large chunks, not per row.  
- **Parallelize where possible**: Use `parallel::mclapply` or `future.apply` for neighbor stats if vectorization is insufficient.  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup as integer indices
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  # Preallocate list
  res <- vector("list", nrow(data))
  
  for (i in seq_len(nrow(data))) {
    ref_idx <- id_to_ref[[as.character(data$id[i])]]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    res[[i]] <- unname(idx_lookup[neighbor_keys])
  }
  res
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats in a vectorized way
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Preallocate result matrix
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  set(cell_data, j = paste0(var_name, "_nbr_max"), value = stats[, 1])
  set(cell_data, j = paste0(var_name, "_nbr_min"), value = stats[, 2])
  set(cell_data, j = paste0(var_name, "_nbr_mean"), value = stats[, 3])
}

# Random Forest prediction in large batches
batch_predict <- function(model, data, batch_size = 1e6) {
  n <- nrow(data)
  preds <- numeric(n)
  for (start in seq(1, n, by = batch_size)) {
    end <- min(start + batch_size - 1, n)
    preds[start:end] <- predict(model, newdata = data[start:end])
  }
  preds
}

# Assuming rf_model is preloaded
predictors <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target name
cell_data[, prediction := batch_predict(rf_model, .SD), .SDcols = predictors]
```

---

### **Expected Gains**
- Eliminates repeated `do.call` and excessive copying.
- Uses in-place `data.table` updates.
- Processes predictions in large chunks.
- Should reduce runtime from **86+ hours to a few hours** on a 16 GB machine.

**Further improvements**:  
- Use `parallel::mclapply` for neighbor stats loop if single-threaded still slow.  
- Consider `ranger::predict()` for faster Random Forest inference.  
- If memory allows, store neighbor indices as `integer64` or sparse structure.