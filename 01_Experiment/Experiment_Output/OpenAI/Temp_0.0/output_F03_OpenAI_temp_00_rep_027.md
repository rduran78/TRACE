 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a full list of length = number of rows (6.46M), which is huge and unnecessary since neighbors depend only on cell IDs, not years.  
4. **Random Forest inference**: If predictions are done in small chunks or with repeated model loading, this adds overhead.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships once per cell**, not per row-year.  
- **Vectorize neighbor feature computation** using `data.table` or `dplyr` joins instead of `lapply`.  
- **Avoid repeated copies**: compute all neighbor stats in one pass and `cbind` results.  
- **Batch predictions**: load the model once, predict on the full dataset or large chunks.  
- **Memory efficiency**: use `data.table` for in-place operations.  

---

**Optimized Workflow**  
1. Convert `cell_data` to `data.table`.  
2. Expand neighbor relationships across years using a join instead of building a 6.46M-element list.  
3. Compute neighbor stats with grouped aggregation.  
4. Add features in one go.  
5. Predict in large batches.  

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors, names = id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand across years by joining with unique years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Join neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in vars) {
  # Join neighbor variable
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(v))]
  setkey(tmp, neighbor_id, year)
  setkey(neighbor_dt, neighbor_id, year)
  neighbor_dt <- tmp[neighbor_dt, on = .(neighbor_id, year)]
  
  # Compute stats by (id, year)
  stats <- neighbor_dt[!is.na(val), .(
    max = max(val, na.rm = TRUE),
    min = min(val, na.rm = TRUE),
    mean = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("id", "year", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  cell_data <- stats[cell_data]
}

# Random Forest prediction
# Load model once
rf_model <- readRDS("trained_rf_model.rds")

# Predict in large batches
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)
for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, gdp_pred := preds]
```

---

**Why this is faster**  
- Eliminates 6.46M `lapply` calls → replaced with vectorized joins and grouped aggregation.  
- Avoids building a massive neighbor list in memory.  
- Uses `data.table` for efficient joins and in-place updates.  
- Predicts in large batches to minimize overhead.  

**Expected improvement**: From 86+ hours to a few hours (depending on disk I/O and CPU).