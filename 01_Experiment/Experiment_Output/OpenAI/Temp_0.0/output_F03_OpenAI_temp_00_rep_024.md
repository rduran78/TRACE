 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-inefficient.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Random Forest inference**: If predictions are done in small chunks or with repeated model loading, this adds overhead.  
4. **Memory pressure**: 6.46M rows × 110+ predictors × multiple neighbor features can exceed 16 GB if not handled efficiently.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships in a long format** (edge list) and join instead of row-wise lookups.  
- **Vectorize neighbor statistics** using `data.table` or `dplyr` group operations instead of `lapply`.  
- **Batch predictions**: Load the Random Forest model once and predict in large chunks.  
- **Avoid repeated copies**: Compute all neighbor features in one pass and merge back.  
- **Use `data.table` for speed and memory efficiency**.  

---

**Optimized Workflow in R**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (id -> neighbor_id)
# rook_neighbors_unique is an nb object: list of integer vectors
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand to panel by joining on year
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_panel, "year", "year")  # ensure year column present

# Merge with cell_data to get neighbor values
# Create a key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_panel, neighbor_id, year)

# For each source var, compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor values
  neighbor_panel[, val := cell_data[.SD, on = .(id = neighbor_id, year), get(var)]]
  
  # Compute stats by (id, year)
  stats <- neighbor_panel[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}

# Random Forest prediction
# Ensure rf_model is loaded once
# Predict in large batches
batch_size <- 1e6
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminates per-row `lapply` and repeated copying.  
- Uses `data.table` joins and grouped aggregation for neighbor stats.  
- Processes predictions in large batches to minimize overhead.  
- Preserves the trained Random Forest model and original estimand.  

This approach should reduce runtime from **86+ hours to a few hours or less**, depending on disk I/O and CPU.