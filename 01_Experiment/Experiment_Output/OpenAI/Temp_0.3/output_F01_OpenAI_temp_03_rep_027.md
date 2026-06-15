 **Diagnosis:**  
- The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`.  
- `joined_data` and `pred_db` are likely large `data.frame`s, causing memory overhead.  
- Predictions are done sequentially without chunking or parallelization.  
- `dplyr` filtering inside the loop repeatedly copies large data, increasing memory usage.  

---

**Optimization Strategy:**  
1. **Convert to `data.table`** for efficient subsetting and assignment.  
2. **Chunk predictions** for large datasets to avoid memory overload.  
3. **Use `parallel::mclapply` or `future.apply`** for parallel predictions by year (if multiple cores available).  
4. **Avoid repeated filtering** by splitting `joined_data` by year once.  
5. **Write results incrementally** to disk to reduce RAM usage.  

---

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, by = "year")

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

# Parallel prediction by year
years <- unique(pred_db$year)
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("rf_models_per_year", "data_by_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, years, function(yr) {
  if (as.character(yr) %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[as.character(yr)]]
    test_set <- data_by_year[[as.character(yr)]]
    preds <- predict_in_chunks(model, test_set)
    list(year = yr, preds = preds)
  } else {
    NULL
  }
})
stopCluster(cl)

# Assign predictions back efficiently
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

**Key Gains:**  
- `data.table` reduces memory copies.  
- Chunking prevents RAM overflow.  
- Parallelization speeds up multi-year prediction.  
- No retraining; models are reused as requested.  

This approach makes the process feasible on a normal machine while preserving trained models.