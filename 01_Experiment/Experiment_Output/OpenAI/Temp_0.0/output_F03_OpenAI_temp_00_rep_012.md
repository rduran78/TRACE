 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists, causing huge overhead in R’s interpreter and memory allocations.  
2. **Inefficient neighbor lookup**: `build_neighbor_lookup` constructs a large list of integer vectors (6.46M elements), which is extremely memory-heavy and slow to traverse repeatedly.  
3. **Repeated feature computation**: For each of 5 variables, `compute_neighbor_stats` loops again over all rows, multiplying the cost.  
4. **Random Forest prediction**: If predictions are done in small chunks or with repeated model loading, this adds overhead.  
5. **Memory pressure**: 6.46M rows × 110+ features is large but manageable if processed in a vectorized or chunked way, not row-wise.  

---

**Optimization Strategy**  
- **Avoid per-row loops**: Use vectorized joins or matrix operations instead of `lapply` over millions of rows.  
- **Precompute neighbor relationships in long format**: Create a data frame of `(cell_id, year, neighbor_id)` and join features once, then aggregate with `data.table` or `dplyr`.  
- **Compute all neighbor stats in one pass**: Instead of looping over variables, melt data and compute grouped summaries.  
- **Use `data.table` for speed and memory efficiency**.  
- **Chunk predictions**: Use `predict(model, newdata, type="response", ...)` in large batches (e.g., 500k rows) to avoid memory spikes.  
- **Keep model in memory**: Load once, reuse.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.table), columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique
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

# Expand for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# 2. Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor values
neighbor_dt <- merge(
  neighbor_dt,
  cell_data[, c("id", "year", vars), with = FALSE],
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  all.x = TRUE
)

# 3. Compute neighbor stats grouped by (id, year)
agg_exprs <- lapply(vars, function(v) {
  list(
    as.name(paste0(v, "_max")) := max(get(v), na.rm = TRUE),
    as.name(paste0(v, "_min")) := min(get(v), na.rm = TRUE),
    as.name(paste0(v, "_mean")) := mean(get(v), na.rm = TRUE)
  )
})

neighbor_stats <- neighbor_dt[, eval(do.call(c, agg_exprs)), by = .(id, year)]

# Replace NaN with NA
for (col in names(neighbor_stats)) {
  set(neighbor_stats, which(is.nan(neighbor_stats[[col]])), col, NA)
}

# 4. Merge back to main data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# 5. Random Forest prediction in chunks
predict_in_chunks <- function(model, data, chunk_size = 500000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ], type = "response")
  }
  preds
}

# Prepare predictor columns (exclude id/year/target)
predictors <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target_var
rf_preds <- predict_in_chunks(rf_model, cell_data[, predictors, with = FALSE])

# Add predictions
cell_data[, prediction := rf_preds]
```

---

**Why this is faster**  
- Eliminates 6.46M-row `lapply` calls and repeated loops.  
- Computes all neighbor stats in one grouped aggregation using `data.table` (highly optimized C backend).  
- Handles all variables in one pass.  
- Uses chunked prediction to avoid memory overflow.  
- Keeps model loaded in memory.  

**Expected improvement**: From 86+ hours to a few hours (or less) on a 16 GB laptop.