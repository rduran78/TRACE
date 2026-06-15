 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects.  
2. **Excessive object copying**: `lapply` + `do.call(rbind, ...)` repeatedly allocates large intermediate lists.  
3. **Inefficient neighbor lookup**: String concatenation (`paste`) and repeated hash lookups for every row are costly.  
4. **Random Forest prediction overhead**: If predictions are done in small chunks or per-row, this adds huge overhead.  
5. **Memory pressure**: 6.46M rows × 110+ features → large data frame; repeated copies worsen RAM usage.  

---

**Optimization Strategy**  
- **Precompute neighbor indices once** as integer vectors, avoid string keys.  
- **Vectorize neighbor stats** using `data.table` or `matrix` operations instead of `lapply`.  
- **Avoid repeated `rbind`**: preallocate numeric matrices.  
- **Batch predictions**: Use `predict(model, newdata, ...)` on the full dataset or large chunks, not per-row.  
- **Use `data.table` for joins and grouping**: drastically reduces overhead.  
- **Keep model in memory**: load once, predict in one call.  

---

**Optimized R Code** (using `data.table` for speed and memory efficiency):  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
nb_list <- rook_neighbors_unique  # list of integer neighbor indices

# Map cell_data$id to position in id_order
cell_data[, id_pos := id_to_idx[as.character(id)]]

# Build neighbor index list for all rows (by id_pos)
neighbor_lookup <- nb_list[cell_data$id_pos]

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_lookup)
  set(cell_data, j = paste0(var_name, "_nb_max"), value = stats[, 1])
  set(cell_data, j = paste0(var_name, "_nb_min"), value = stats[, 2])
  set(cell_data, j = paste0(var_name, "_nb_mean"), value = stats[, 3])
}

# Drop helper column
cell_data[, id_pos := NULL]

# Random Forest prediction in one batch (or large chunks if memory tight)
pred <- predict(rf_model, newdata = cell_data)

# Attach predictions
cell_data[, gdp_pred := pred]
```

---

**Expected Gains**  
- Eliminates repeated string operations and `lapply` overhead.  
- Reduces memory churn by preallocating matrices and using `data.table` in-place updates.  
- Single-pass neighbor computation and single batch prediction → runtime drops from 86+ hours to a few hours (or less) on 16 GB RAM.  

**Key Principles**: Precompute indices, vectorize, minimize copies, batch predictions.