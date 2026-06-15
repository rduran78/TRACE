 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a full list of neighbors for every row, even though neighbors are static across years.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is catastrophic. `predict()` should be vectorized on the full data frame or large chunks.  
5. **Memory pressure**: 6.46M rows × 110+ features is large but manageable if processed in chunks and without redundant copies.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships once per cell** (not per cell-year).  
- **Use vectorized joins instead of nested `lapply`**: Reshape data so neighbor stats are computed via `data.table` or `dplyr` group operations.  
- **Chunked processing**: Compute features and predictions in chunks to avoid RAM exhaustion.  
- **Avoid repeated model loading**: Load the Random Forest model once.  
- **Vectorized prediction**: Call `predict(rf_model, newdata = chunk)` on large batches, not per row.  

---

**Optimized Workflow in R (using `data.table`)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

setDT(cell_data)  # convert to data.table
# cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.

# Precompute neighbor pairs (cell_id -> neighbor_id)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand neighbor pairs across years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# Merge neighbor values for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  neighbor_pairs <- merge(neighbor_pairs, tmp, by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Compute stats by (id, year)
  stats <- neighbor_pairs[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Rename and merge back
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
  
  # Drop val column from neighbor_pairs to save memory
  neighbor_pairs[, val := NULL]
}

# At this point, cell_data has all neighbor features added

# Random Forest prediction in chunks
chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminates per-row `lapply` → replaced with vectorized `data.table` joins and aggregations.  
- Avoids repeated copying of `cell_data`.  
- Processes predictions in large chunks instead of row-by-row.  
- Memory footprint reduced by dropping intermediate columns after use.  

**Expected runtime**: From 86+ hours to a few hours (or less) on a 16 GB laptop.