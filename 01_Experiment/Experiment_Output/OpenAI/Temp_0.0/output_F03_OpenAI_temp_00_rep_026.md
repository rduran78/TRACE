 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies the entire `data.frame` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a large list of integer vectors for every row, which is costly in both time and memory.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is catastrophic. `predict()` on the entire data frame at once is essential.  

---

**Optimization Strategy**  
- **Vectorize neighbor feature computation**: Instead of looping over rows, use `data.table` joins or matrix aggregation.  
- **Precompute neighbor relationships as a long table**: `(cell_id, year, neighbor_id)` and join to compute stats in bulk.  
- **Avoid repeated copies**: Use `data.table` for in-place updates.  
- **Batch Random Forest predictions**: Call `predict(rf_model, newdata)` once on the full dataset or in large chunks.  
- **Memory efficiency**: Drop intermediate objects early, use integer keys, and avoid large nested lists.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded
setDT(cell_data)  # convert to data.table
setkey(cell_data, id, year)

# Precompute neighbor pairs as a long table
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Join neighbor values for each variable and compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
               by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Compute stats by (id, year)
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                               paste0(var_name, "_nbr_min"),
                               paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# Drop neighbor_dt and tmp to free memory
rm(neighbor_dt, tmp); gc()

# Random Forest prediction in batch
# Ensure predictor columns match model
predictors <- setdiff(names(cell_data), c("id", "year", "target_var"))  # adjust target_var
preds <- predict(rf_model, newdata = cell_data[, ..predictors])

# Attach predictions
cell_data[, gdp_pred := preds]
```

---

**Expected Gains**  
- Eliminates 6.46M-row `lapply` loops → replaced with vectorized `data.table` aggregation.  
- Avoids repeated full data copies.  
- Random Forest predictions done in one call.  
- Should reduce runtime from **86+ hours to a few hours or less**, depending on I/O and CPU.  

**Key Notes**  
- Ensure `rf_model` and `cell_data` predictor columns align.  
- If memory is still tight, process neighbor stats and predictions in yearly chunks.  
- This preserves the trained model and original estimand.