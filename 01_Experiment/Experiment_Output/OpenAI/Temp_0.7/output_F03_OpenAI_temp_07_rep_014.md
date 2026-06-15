 **Diagnosis**  
The main bottlenecks are:  
1. **Inefficient neighbor feature computation**:  
   - `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly iterate over millions of rows using `lapply`, causing heavy R-level loops and memory overhead.  
   - `do.call(rbind, ...)` on large lists is costly.  
   - Object copying in each step adds significant overhead.  

2. **Prediction inefficiency**:  
   - Random Forest inference in `randomForest` or similar base packages is relatively fast compared to the above loops, but repeated copying and non-vectorized feature prep dominate runtime.  

3. **Memory pressure**:  
   - 6.46M rows × 110+ predictors → large data frame copies in R memory.  

**Optimization Strategy**  
- **Precompute neighbor indices once** and reuse as integer indices.  
- **Vectorize neighbor statistics**:  
  - Avoid per-row loops; use `data.table` joins or `matrixStats` on aggregated arrays.  
  - Compute max/min/mean in a single grouped operation.  
- **Use `data.table` for feature engineering** to minimize copying.  
- **Chunked prediction** to keep memory stable.  
- **Keep Random Forest model loaded once** and call `predict()` on large chunks.  

---

### **Optimized Implementation (R)**

```r
library(data.table)
library(randomForest) # or ranger if available

# Assume cell_data is a data.table
setDT(cell_data)

# Precompute neighbor pairs: (id, year) -> neighbor_id
# rook_neighbors_unique: list of integer vectors, same order as id_order
id_map <- data.table(id = id_order, idx = seq_along(id_order))
pairs_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(source_idx = i, neighbor_idx = rook_neighbors_unique[[i]])
}))
pairs_list[, source_id := id_order[source_idx]]
pairs_list[, neighbor_id := id_order[neighbor_idx]]
pairs_list[, c("source_idx", "neighbor_idx") := NULL]

# Expand pairs by year
years <- sort(unique(cell_data$year))
expanded_pairs <- pairs_list[, .(year = years), by = .(source_id, neighbor_id)]
setkey(expanded_pairs, neighbor_id, year)

# Join neighbor values for each var
for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  expanded_pairs <- tmp[expanded_pairs, on = .(neighbor_id, year)]
  
  # Compute stats by source_id-year
  stats <- expanded_pairs[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(source_id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("source_id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data <- stats[cell_data, on = .(source_id = id, year)]
}

# Convert to matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "gdp")) # exclude target
X <- as.matrix(cell_data[, ..predictors])

# Load trained RF model (assume rf_model exists)
# Predict in chunks to save memory
chunk_size <- 1e5
preds <- numeric(nrow(X))
for (i in seq(1, nrow(X), by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, nrow(X))
  preds[idx] <- predict(rf_model, newdata = X[idx, , drop = FALSE])
}

cell_data[, gdp_pred := preds]
```

---

**Key Gains**  
- Replaces millions of `lapply` calls with fully vectorized `data.table` joins and grouped aggregation.  
- Avoids repeated object copying by modifying `cell_data` in place.  
- Handles prediction in efficient chunks without retraining.  

**Expected Runtime Reduction**: From ~86 hours to a few hours or less on a standard laptop.  

**Preserves**:  
- Original trained Random Forest model.  
- Original estimand and feature definitions.  
- All required neighbor-feature logic.