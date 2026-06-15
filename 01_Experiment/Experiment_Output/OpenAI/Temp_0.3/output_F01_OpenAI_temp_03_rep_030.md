 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets with `predict()`. Each iteration loads large subsets into memory, and `predict()` on large data frames is memory-intensive. Additionally, `dplyr` operations and repeated subsetting add overhead.

---

**Optimization Strategy:**  
1. **Avoid repeated filtering:** Split data once by year instead of filtering in each iteration.  
2. **Use `data.table` for memory efficiency:** Faster subsetting and assignment.  
3. **Predict in chunks:** Process large datasets in smaller batches to reduce memory usage.  
4. **Parallelize predictions:** Use multiple cores via `parallel::mclapply` or `future.apply`.  
5. **Write intermediate results:** Avoid holding the entire prediction in memory if not needed.  

---

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, joined_data$year)

# Function to predict in chunks
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  preds
}

# Parallel prediction
years <- names(data_by_year)
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("rf_models_per_year", "data_by_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, years, function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[yr]]
    dt <- data_by_year[[yr]]
    preds <- predict_in_chunks(model, dt)
    list(year = yr, preds = preds, rows = dt[, .I])
  } else {
    NULL
  }
})
stopCluster(cl)

# Assign predictions back
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == as.integer(res$year), consolidated := res$preds]
  }
}

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Gains:**  
- `data.table` reduces memory overhead.  
- Chunked prediction prevents memory overflow.  
- Parallelization uses multiple cores for speed.  
- Preserves trained models without retraining.  

This approach makes the process feasible on a normal machine while handling large datasets efficiently.