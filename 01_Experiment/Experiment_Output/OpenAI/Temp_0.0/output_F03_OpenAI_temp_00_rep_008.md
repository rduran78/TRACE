 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists, causing huge overhead in R’s interpreter and memory allocations.  
2. **Inefficient neighbor lookup**: `build_neighbor_lookup` constructs a large list of integer vectors (6.46M elements), which is extremely memory-heavy.  
3. **Repeated feature computation**: For each of 5 variables, `compute_neighbor_stats` loops again over all rows, multiplying the cost.  
4. **Random Forest inference**: If predictions are done in small chunks or with repeated model loading, this adds overhead.  
5. **Memory pressure**: 6.46M rows × 110 predictors + neighbor features is large; copying data frames repeatedly is expensive.  

---

**Optimization Strategy**  
- **Avoid per-row `lapply`**: Use vectorized or matrix-based operations.  
- **Precompute neighbor relationships in long format**: Instead of storing a list of neighbors per row, create a data frame of `(cell_id, year, neighbor_id)` and join.  
- **Compute neighbor stats with `data.table`**: Aggregate max, min, mean in one grouped operation.  
- **Chunked prediction**: Use `predict()` on large chunks (e.g., 500k rows) to avoid memory spikes.  
- **Keep model in memory**: Load once, reuse.  
- **Use `data.table` for all joins and aggregations** for speed and memory efficiency.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.table), columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell ids in same order as rook_neighbors_unique
# rf_model: pre-trained randomForest object

setDT(cell_data)
setkey(cell_data, id, year)

# 1. Build neighbor pairs in long format
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# 2. Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# 3. Join neighbor values and compute stats for all vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor values
  tmp <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var))],
               by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Compute stats by (id, year)
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("id", "year", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# 4. Random Forest prediction in chunks
predictors <- setdiff(names(cell_data), c("id", "year"))  # exclude keys
chunk_size <- 500000
n <- nrow(cell_data)
pred <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  pred[idx] <- predict(rf_model, newdata = cell_data[idx, ..predictors])
}

cell_data[, gdp_pred := pred]
```

---

**Why this is faster**  
- Eliminates 6.46M `lapply` calls → replaced with vectorized `data.table` joins and aggregations.  
- Computes neighbor stats in **5 grouped operations** instead of 5 × 6.46M loops.  
- Avoids building a massive list of neighbor indices.  
- Uses chunked prediction to keep memory stable.  

**Expected improvement**: From 86+ hours to a few hours (depending on disk I/O and CPU).