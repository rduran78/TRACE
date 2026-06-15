 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large data frames with `predict()`. Each iteration filters `joined_data` and writes predictions into `pred_db`, which is inefficient for hundreds of thousands of rows. Memory overhead is high due to repeated subsetting and copying.

---

### **Optimization Strategy**
1. **Avoid repeated filtering**: Split `joined_data` by year once instead of filtering inside the loop.
2. **Use `data.table` for fast subsetting and assignment**.
3. **Predict in chunks**: If memory is limited, process large datasets in manageable batches.
4. **Preallocate prediction column**: Already done, but use `data.table` for efficiency.
5. **Parallelize predictions**: Use `parallel::mclapply` or `future.apply` for multi-core prediction.
6. **Keep models unchanged**: Only optimize data handling and prediction steps.

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
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

# Parallel prediction across years
years <- names(data_by_year)
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("rf_models_per_year", "data_by_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, years, function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[yr]]
    dt <- data_by_year[[yr]]
    preds <- predict_in_chunks(model, dt)
    list(year = as.integer(yr), preds = preds, rows = dt[, .I])
  } else {
    NULL
  }
})
stopCluster(cl)

# Assign predictions back
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Why This Works**
- **`data.table`** drastically reduces filtering and assignment overhead.
- **Chunked prediction** prevents memory overload.
- **Parallelization** uses multiple cores for speed.
- **No retraining**: Models remain unchanged.

This approach should make the process feasible on a normal machine while preserving accuracy.