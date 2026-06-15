 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows** in `build_neighbor_lookup` and `compute_neighbor_stats`. These create millions of small vectors and heavy object copying.  
2. **Neighbor feature computation** is done in R loops rather than vectorized or matrix-based operations.  
3. **Random Forest inference** is likely slowed by repeated model loading and prediction on small chunks instead of batch prediction.  
4. **Memory overhead** from large lists and repeated `do.call(rbind, …)` calls.  

---

**Optimization Strategy**  
- **Precompute neighbor index matrix once** as an integer matrix (rows = cells, cols = neighbors) to avoid millions of list elements.  
- **Use `data.table` or `matrixStats`** for fast aggregation (max, min, mean) across neighbors.  
- **Vectorize neighbor feature computation**: operate on entire columns at once instead of looping per row.  
- **Batch Random Forest predictions**: load model once, predict on large chunks or full dataset.  
- **Avoid repeated object copying**: modify in place with `data.table`.  

---

**Optimized R Code**  

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor index matrix (cells x max_neighbors)
build_neighbor_matrix <- function(id_order, neighbors) {
  max_n <- max(lengths(neighbors))
  mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_n)
  for (i in seq_along(neighbors)) {
    nbs <- neighbors[[i]]
    if (length(nbs) > 0) {
      mat[i, seq_along(nbs)] <- nbs
    }
  }
  mat
}

neighbor_mat <- build_neighbor_matrix(id_order, rook_neighbors_unique)

# Map id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Compute neighbor stats for all years efficiently
compute_neighbor_features <- function(dt, var_name, neighbor_mat, id_to_idx) {
  vals <- dt[[var_name]]
  n_cells <- length(id_order)
  n_years <- length(unique(dt$year))
  
  # Reshape data to matrix: rows = cells, cols = years
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  year_levels <- sort(unique(dt$year))
  for (i in seq_along(year_levels)) {
    yr <- year_levels[i]
    idx <- dt$year == yr
    val_mat[id_to_idx[as.character(dt$id[idx])], i] <- vals[idx]
  }
  
  # Compute neighbor stats per cell-year
  max_mat <- min_mat <- mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbor_mat[i, ]
    nb_idx <- nb_idx[!is.na(nb_idx)]
    if (length(nb_idx) > 0) {
      nb_vals <- val_mat[nb_idx, , drop = FALSE]
      max_mat[i, ]  <- colMaxs(nb_vals, na.rm = TRUE)
      min_mat[i, ]  <- colMins(nb_vals, na.rm = TRUE)
      mean_mat[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
    }
  }
  
  # Melt back to long and join
  res <- data.table(
    id = rep(id_order, times = n_years),
    year = rep(year_levels, each = n_cells),
    paste0(var_name, "_nb_max") = as.vector(max_mat),
    paste0(var_name, "_nb_min") = as.vector(min_mat),
    paste0(var_name, "_nb_mean") = as.vector(mean_mat)
  )
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features and merge
neighbor_features <- NULL
for (var_name in neighbor_source_vars) {
  feat <- compute_neighbor_features(cell_data, var_name, neighbor_mat, id_to_idx)
  neighbor_features <- if (is.null(neighbor_features)) feat else merge(neighbor_features, feat, by = c("id", "year"))
}

# Merge with original data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"))

# Random Forest prediction
# Load model once
rf_model <- readRDS("trained_rf_model.rds")

# Predict in large batches
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)
for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, gdp_pred := preds]
```

---

**Expected Gains**  
- Eliminates millions of small `lapply` calls → replaced with matrix operations.  
- Reduces memory overhead by using dense matrices and `data.table`.  
- Batch prediction avoids repeated model loading.  
- Should cut runtime from 86+ hours to a few hours (depending on I/O and CPU).