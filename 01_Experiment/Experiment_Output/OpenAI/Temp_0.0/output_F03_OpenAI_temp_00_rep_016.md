 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` call copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a large list of integer vectors for every row, which is costly in memory and time.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is catastrophic. `predict()` should be vectorized over the entire dataset or large chunks.  
5. **Memory pressure**: 6.46M rows × 110+ features is large but manageable if processed in chunks and without redundant copies.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships in a long format** (edge list) and join instead of building a huge list.  
- **Compute neighbor stats using `data.table` or `dplyr` group operations** instead of millions of `lapply` calls.  
- **Avoid repeated copies**: Add all neighbor features in one pass.  
- **Batch predictions**: Use `predict()` on the full data or in chunks (e.g., 500k rows per batch).  
- **Keep model in memory**: Load once, reuse.  
- **Use `data.table` for speed and memory efficiency**.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)

# Precompute neighbor edges in long format
# rook_neighbors_unique: list of integer vectors (spdep nb object)
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand edges for all years
years <- unique(cell_data$year)
edges <- edges[, .(id = from, neighbor_id = to), by = years]
setnames(edges, "years", "year")

# Merge neighbor values for each variable and compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- merge(edges,
               cell_data[, .(neighbor_id = id, year, val = get(var_name))],
               by = c("neighbor_id", "year"),
               all.x = TRUE)
  
  # Compute stats by (id, year)
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to main data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# Prepare predictor matrix
predictors <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target_var
X <- cell_data[, ..predictors]

# Batch prediction to avoid memory issues
batch_size <- 500000
n <- nrow(X)
preds <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = X[idx])
}

cell_data[, prediction := preds]
```

---

**Why this is faster**  
- Eliminates millions of `lapply` calls → replaced with vectorized `data.table` joins and aggregations.  
- Avoids repeated copying of `cell_data`.  
- Computes all neighbor stats in grouped operations.  
- Predicts in large batches instead of row-by-row.  

**Expected improvement**: From 86+ hours to a few hours (or less) on a 16 GB laptop.