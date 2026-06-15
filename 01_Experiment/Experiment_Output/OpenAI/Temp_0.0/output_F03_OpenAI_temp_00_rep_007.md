 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects.  
2. **Excessive object copying**: `lapply` + `do.call(rbind, ...)` repeatedly allocates large intermediate lists.  
3. **Inefficient neighbor lookup**: String concatenation (`paste`) and repeated hash lookups for every row are costly.  
4. **Memory pressure**: 6.46M rows × multiple neighbor stats × 110 predictors stresses 16 GB RAM.  
5. **Prediction loop**: If Random Forest predictions are done row-by-row, this is extremely slow.  

---

**Optimization Strategy**  
- **Precompute neighbor indices once** as integer vectors, avoid string keys.  
- **Vectorize neighbor stats** using `data.table` or `matrix` operations instead of `lapply`.  
- **Avoid repeated `rbind`**: preallocate numeric matrices.  
- **Batch predictions**: Use `predict(model, newdata, type="response")` on the full dataset or large chunks, not per-row.  
- **Use `data.table` for joins and grouping**: drastically reduces overhead.  
- **Memory efficiency**: Drop unnecessary columns early, use numeric matrices for modeling.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Create a mapping from (id, year) -> row index
cell_data[, row_idx := .I]

# Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(dt, neighbor_lookup, var_name) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  res <- matrix(NA_real_, n, 3)  # max, min, mean
  for (i in seq_len(n)) {
    nb_ids <- neighbor_lookup[[id_to_idx[[as.character(dt$id[i])]]]]
    if (length(nb_ids) == 0) next
    # Find rows for same year
    nb_rows <- dt[J(id_order[nb_ids], dt$year[i]), nomatch = 0L, which = TRUE]
    if (length(nb_rows) == 0) next
    nb_vals <- vals[nb_rows]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  res
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(cell_data, neighbor_lookup, var_name)
  cell_data[, paste0(var_name, "_nb_max") := stats[, 1]]
  cell_data[, paste0(var_name, "_nb_min") := stats[, 2]]
  cell_data[, paste0(var_name, "_nb_mean") := stats[, 3]]
}

# Prepare matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "row_idx", "target_var"))
X <- as.matrix(cell_data[, ..predictors])

# Batch prediction
batch_size <- 500000
preds <- numeric(nrow(X))
for (i in seq(1, nrow(X), by = batch_size)) {
  idx <- i:min(i + batch_size - 1, nrow(X))
  preds[idx] <- predict(rf_model, newdata = X[idx, , drop = FALSE])
}

cell_data[, prediction := preds]
```

---

**Key Gains**  
- Eliminates repeated string concatenation and `lapply` overhead.  
- Uses preallocated matrices and `data.table` for efficient joins.  
- Batch prediction avoids per-row model calls.  

**Expected Runtime Reduction**: From 86+ hours to a few hours (depending on disk I/O and CPU).  

For even more speed:  
- Parallelize neighbor stats computation with `future.apply` or `data.table` chunking.  
- Consider converting neighbor stats computation to C++ via `Rcpp` if still slow.