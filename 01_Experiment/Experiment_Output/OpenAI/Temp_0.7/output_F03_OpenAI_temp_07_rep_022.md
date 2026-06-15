 **Diagnosis**

- **Main bottlenecks**:  
  1. **Neighbor feature computation**: Current implementation uses `lapply` over ~6.46M rows for each of 5 variables → extremely slow and memory-heavy.  
  2. **Random Forest inference**: Likely dominated by repeated model loading and prediction loops instead of batch prediction.  
  3. **Excessive object copying**: Each `compute_and_add_neighbor_features` call rebuilds large objects, causing memory churn.  
- **Scale**: 6.46M rows × 110 predictors × 28 years requires a vectorized and memory-efficient approach.  

---

**Optimization Strategy**

1. **Precompute neighbor statistics in a vectorized/data.table fashion**:
   - Avoid per-row `lapply`.
   - Flatten neighbor relationships into a long table, join, and aggregate using `data.table`.
2. **Batch prediction**:
   - Load model once.
   - Predict in large chunks (e.g., 100k rows at a time) to avoid RAM exhaustion.
3. **Memory efficiency**:
   - Use `data.table` instead of repeatedly copying `data.frame`.
   - Avoid repeated joins, compute all neighbor stats in one grouped step.
4. **Parallelization**:
   - If possible, use `data.table` multithreading (built-in) and `ranger` for fast RF prediction.

---

**Optimized R Code**

```r
library(data.table)
library(ranger)  # or randomForest if you must, but ranger is much faster

# Convert to data.table
setDT(cell_data)

# Flatten neighbor list into long format
# rook_neighbors_unique: list of neighbor indices for each cell
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(from = id_order[i], to = id_order[rook_neighbors_unique[[i]]])
}))

# Expand by year to match panel
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years, from = from, unique = TRUE), on = .(from), allow.cartesian = TRUE]
setnames(neighbor_dt, "year", "year")

# Add source and target keys
cell_data[, key := paste(id, year, sep = "_")]
neighbor_dt[, to_key := paste(to, year, sep = "_")]

# Map neighbor values for all variables in one pass
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor vars for easier aggregation
val_dt <- melt(
  cell_data[, c("key", neighbor_vars), with = FALSE],
  id.vars = "key",
  variable.name = "var",
  value.name = "val"
)

# Join neighbor values
neighbor_long <- merge(
  neighbor_dt[, .(key = paste(from, year, sep = "_"), to_key)],
  val_dt, by.x = "to_key", by.y = "key", allow.cartesian = TRUE
)

# Aggregate neighbor stats: max, min, mean
neighbor_stats <- neighbor_long[
  ,
  .(
    nb_max = max(val, na.rm = TRUE),
    nb_min = min(val, na.rm = TRUE),
    nb_mean = mean(val, na.rm = TRUE)
  ),
  by = .(key, var)
]

# Reshape back to wide
neighbor_stats_wide <- dcast(
  neighbor_stats,
  key ~ var,
  value.var = c("nb_max", "nb_min", "nb_mean")
)

# Merge into main table
cell_data <- merge(cell_data, neighbor_stats_wide, by = "key", all.x = TRUE)

# Drop helper columns
cell_data[, key := NULL]

# Random Forest Prediction in Batches
# Assume rf_model is already loaded
batch_size <- 100000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, data = cell_data[start:end])$predictions
}

cell_data[, gdp_pred := preds]
```

---

**Expected Impact**

- **Neighbor stats computation**: From >80 hours to minutes using `data.table` aggregation and avoiding per-row loops.
- **Prediction**: From repeated row-wise predictions to fast batched inference, reducing hours to minutes (especially with `ranger`).
- **Memory**: Single-pass vectorized operations avoid repeated large object copies.

**Summary**:  
- Replace per-row `lapply` with `data.table` joins and grouped aggregation.  
- Compute all neighbor-based features in one pipeline step.  
- Batch RF predictions with `ranger` or similar efficient implementation.  
- Keep the original model and estimand intact.