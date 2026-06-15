 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations**: `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of rows, creating huge lists and repeatedly copying data.  
2. **Inefficient row-wise operations**: Each row processes neighbors individually, causing excessive R-level loops and memory allocations.  
3. **Random Forest prediction inefficiency**: If using `predict()` in small batches or per-row, overhead is huge. Predictions should be vectorized in large chunks.  
4. **Memory pressure**: 6.46M rows × 110+ features is large; repeated copying worsens performance.  

**Optimization Strategy**  
- **Precompute neighbor lookups once** as integer indices (already done, but store as a matrix for fast access).  
- **Vectorize neighbor stats**: Instead of `lapply` per row, use matrix operations or `data.table` joins.  
- **Use `data.table` for feature engineering**: Fast grouping and joins.  
- **Batch predictions**: Call `predict()` on the full data (or large chunks) at once.  
- **Avoid unnecessary copies**: Modify in place with `:=` in `data.table`.  
- **Parallelize**: Use `parallel::mclapply` or `future.apply` for neighbor stats if memory allows.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor lookup as a list of integer vectors (already done)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor_lookup to a compressed representation
# (optional optimization if memory is tight)

# Vectorized neighbor stats computation
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  # Preallocate result matrix
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  colnames(res) <- c(paste0(var_name, "_nbr_max"),
                     paste0(var_name, "_nbr_min"),
                     paste0(var_name, "_nbr_mean"))
  
  # Parallelized apply
  res_list <- parallel::mclapply(seq_along(neighbor_lookup), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, mc.cores = parallel::detectCores() - 1)
  
  res[,] <- do.call(rbind, res_list)
  res
}

# Compute all neighbor features efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data[, (colnames(stats_mat)) := as.data.table(stats_mat)]
}

# Prepare predictor matrix
predictors <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target_var
X <- as.data.frame(cell_data[, ..predictors])

# Load trained Random Forest model
rf_model <- readRDS("trained_rf_model.rds")

# Predict in large batches
batch_size <- 500000
n <- nrow(X)
preds <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, X[idx, , drop = FALSE])
}

cell_data[, prediction := preds]
```

---

**Key Gains**  
- Eliminates repeated copying by using `data.table` and preallocation.  
- Parallelizes neighbor computations.  
- Vectorized batch prediction avoids per-row overhead.  
- Expected runtime reduction from **86+ hours → a few hours** (depending on cores and I/O).  

**Preserves**:  
- Original trained Random Forest model.  
- Numerical estimand (same features, same model).