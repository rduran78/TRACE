 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- `joined_data` is filtered repeatedly, creating large intermediate objects.
- Predictions are done sequentially without leveraging efficient data handling.
- `pred_db` is updated in a slow vectorized way inside the loop.

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment.
- Avoid repeated subsetting by splitting data once.
- Use `predict()` in chunks to reduce memory load.
- Parallelize predictions with `parallel` or `future.apply`.
- Write results incrementally to disk to avoid holding everything in memory.

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, by = "year")

# Detect available cores for parallel processing
n_cores <- max(1, detectCores() - 1)

# Function to predict in chunks
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = newdata[idx[[i]], ])
  }
  preds
}

# Parallel prediction
cl <- makeCluster(n_cores)
clusterExport(cl, c("rf_models_per_year", "data_by_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, names(data_by_year), function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[yr]]
    dt <- data_by_year[[yr]]
    preds <- predict_in_chunks(model, dt)
    data.table(year = as.integer(yr), id = dt$id, consolidated = preds)
  } else {
    NULL
  }
})
stopCluster(cl)

# Combine results and merge back
pred_results <- rbindlist(results, use.names = TRUE, fill = TRUE)
pred_db[pred_results, on = .(year, id), consolidated := i.consolidated]

# Write to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**
- **`data.table`** for fast filtering and assignment.
- **Chunked prediction** to avoid memory overload.
- **Parallelization** to utilize multiple cores.
- **Incremental merging** instead of updating inside a loop.

This approach preserves the trained models, avoids retraining, and makes the process feasible on a normal machine.