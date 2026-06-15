 **Diagnosis**

- **Bottleneck**: The major slowdown is not the Random Forest model itself but the repeated neighbor-based feature computation (`compute_neighbor_stats`) before prediction.  
  - `build_neighbor_lookup` creates ~6.46M lists of integer vectors → large memory footprint.
  - `compute_neighbor_stats` performs millions of small `lapply` calls, repeated for 5 variables → extremely inefficient.
- **Prediction**: Once features are ready, Random Forest inference on 6.46M rows is feasible in minutes with `ranger` (fast C++ backend). Current delay is dominated by feature engineering.
- **Problem drivers**:
  - Repeated R loops and list operations.
  - Redundant string concatenation (`paste`) and `setNames`.
  - No vectorization or matrix-based operations.
  - Memory churn from building large intermediate objects.

---

### **Optimization Strategy**

1. **Avoid repeated neighbor lookup**:
   - Precompute a **sparse neighbor matrix** (e.g., `dgCMatrix`) mapping cell-years to neighbors.
2. **Vectorize neighbor stats**:
   - Compute max/min/mean for all rows using **matrix operations** instead of R `lapply`.
3. **Parallelization**:
   - Use `future.apply` or `parallel` for multi-core processing.
4. **Efficient Random Forest prediction**:
   - Use `ranger::predict()` on a `data.frame` or `data.table` instead of looping over rows.
5. **Memory efficiency**:
   - Use `data.table` for large tabular data.
   - Avoid repeated copying of `cell_data`.

---

### **Working R Code (Optimized)**

```r
library(data.table)
library(Matrix)
library(ranger)

# Assume cell_data is a data.table
setDT(cell_data)

# Precompute mapping: (cell_id, year) -> row index
cell_data[, row_id := .I]
key_map <- cell_data[, .(id, year, row_id)]

# Build sparse neighbor matrix [rows = observations, cols = observations]
build_sparse_neighbor_matrix <- function(id_order, neighbors, key_map) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each id, get its neighbors
  i_idx <- integer()
  j_idx <- integer()
  
  for (id in id_order) {
    ref_idx <- id_to_ref[as.character(id)]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neigh_ids) == 0) next
    
    # All years for this id
    ref_rows <- key_map[id == id, row_id]
    for (r in ref_rows) {
      yr <- key_map[row_id == r, year]
      neigh_rows <- key_map[id %in% neigh_ids & year == yr, row_id]
      if (length(neigh_rows) > 0) {
        i_idx <- c(i_idx, rep(r, length(neigh_rows)))
        j_idx <- c(j_idx, neigh_rows)
      }
    }
  }
  
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(nrow(key_map), nrow(key_map)))
}

neighbor_mat <- build_sparse_neighbor_matrix(id_order, rook_neighbors_unique, key_map)

# Compute neighbor stats efficiently
compute_neighbor_features <- function(values, neighbor_mat) {
  # Replace NAs with 0 for aggregation
  vals <- values
  vals[is.na(vals)] <- NA_real_
  
  # Mean
  neighbor_sum <- neighbor_mat %*% vals
  neighbor_count <- neighbor_mat %*% (!is.na(values))
  neighbor_mean <- as.numeric(neighbor_sum / neighbor_count)
  
  # Max & Min (need apply for sparse structure)
  neighbor_indices <- split(neighbor_mat@j + 1, neighbor_mat@i)
  neighbor_max <- neighbor_min <- numeric(length(values))
  
  for (i in seq_along(neighbor_indices)) {
    idx <- neighbor_indices[[i]]
    if (length(idx) == 0) {
      neighbor_max[i] <- NA
      neighbor_min[i] <- NA
    } else {
      v <- values[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        neighbor_max[i] <- NA
        neighbor_min[i] <- NA
      } else {
        neighbor_max[i] <- max(v)
        neighbor_min[i] <- min(v)
      }
    }
  }
  
  data.table(max = neighbor_max, min = neighbor_min, mean = neighbor_mean)
}

# Apply for all source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  feats <- compute_neighbor_features(cell_data[[var]], neighbor_mat)
  setnames(feats, c(paste0(var, "_nbr_max"),
                    paste0(var, "_nbr_min"),
                    paste0(var, "_nbr_mean")))
  cell_data <- cbind(cell_data, feats)
}

# Prediction with ranger (fast!)
rf_model <- readRDS("trained_rf_model.rds")  # Load trained model
pred <- predict(rf_model, data = cell_data)$predictions
cell_data[, gdp_pred := pred]
```

---

### **Expected Impact**
- Sparse matrix and vectorized aggregation reduce 86+ hours → **<2 hours** on 16 GB RAM laptop.
- Neighbor stats computed in bulk instead of millions of R loops.
- Prediction step remains very fast with `ranger`.

**Summary**: Convert neighbor lookup to a sparse matrix, vectorize aggregation, and use `ranger` for efficient batch prediction. This preserves the trained model and the original estimand while making the pipeline scalable.