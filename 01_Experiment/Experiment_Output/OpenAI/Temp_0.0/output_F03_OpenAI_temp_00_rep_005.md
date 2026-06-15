 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a full list of neighbor indices for every row, even though neighbors are static across years.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is very slow. `predict()` on the full data frame at once is much faster.  
5. **Memory pressure**: 6.46M rows × 110+ variables is large; repeated intermediate objects exacerbate RAM usage.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships once per cell** (not per cell-year).  
- **Vectorize neighbor feature computation** using `data.table` or `dplyr` joins instead of `lapply`.  
- **Avoid repeated copies**: compute all neighbor stats in one pass and `cbind` results.  
- **Batch Random Forest predictions**: use `predict(rf_model, newdata)` on the entire dataset or in large chunks.  
- **Use `data.table` for speed and memory efficiency**.  

---

**Optimized R Code**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)

# Precompute neighbor relationships (static across years)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years by joining on year
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years, id = id, unique = TRUE), on = .(id)]
setnames(neighbor_dt, "id", "cell_id")
neighbor_dt[, key := paste(cell_id, year, sep = "_")]
neighbor_dt[, neighbor_key := paste(neighbor_id, year, sep = "_")]

# Create lookup for neighbor values
cell_data[, key := paste(id, year, sep = "_")]
setkey(cell_data, key)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  tmp <- neighbor_dt[cell_data, on = .(neighbor_key = key), nomatch = 0]
  # tmp now has neighbor values for each cell-year
  tmp[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(key)]
}

# Compute all neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_dt(var_name)
  setnames(stats_dt, c("key", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data <- stats_dt[cell_data, on = .(key)]
}

# Drop helper columns
cell_data[, key := NULL]

# Random Forest prediction in batch
# Ensure predictor columns match model
predictors <- names(rf_model$forest$xlevels)
predictions <- predict(rf_model, newdata = cell_data[, ..predictors])

# Add predictions
cell_data[, gdp_pred := predictions]
```

---

**Why this is faster**  
- Eliminates per-row `lapply` loops → replaced with vectorized `data.table` joins and grouped aggregation.  
- Neighbor relationships computed once, reused for all variables.  
- Random Forest predictions done in one call, not row-by-row.  
- Memory footprint reduced by avoiding repeated copies of `cell_data`.  

**Expected improvement**: From 86+ hours to a few hours or less on a 16 GB laptop, depending on I/O and disk speed.  

**Preserves**: Trained RF model, original estimand, and neighbor-based feature logic.