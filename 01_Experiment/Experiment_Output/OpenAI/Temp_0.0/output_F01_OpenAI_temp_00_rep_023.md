 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated indexing. This approach is memory-inefficient and slow because:  
- `filter()` and subsetting inside the loop repeatedly copy large data frames.  
- `predict()` is called on large chunks without batching or parallelization.  
- `pred_db$consolidated[...] <- ...` repeatedly modifies a large object in memory.  

**Optimization Strategy:**  
- Use `data.table` for fast filtering and in-place updates.  
- Avoid repeated subsetting by splitting data once.  
- Use `predict()` in chunks to reduce memory load.  
- Optionally enable parallel prediction with `parallel::mclapply` or `future.apply`.  
- Keep models as-is (no retraining).  

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, joined_data$year)

# Function for chunked prediction
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
years <- intersect(names(rf_models_per_year), names(data_by_year))

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("rf_models_per_year", "data_by_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, years, function(yr) {
  model <- rf_models_per_year[[yr]]
  dt <- data_by_year[[yr]]
  preds <- predict_in_chunks(model, dt)
  list(year = as.integer(yr), preds = preds, ids = dt$id) # assuming 'id' exists
})
stopCluster(cl)

# Merge predictions back into pred_db
for (res in results) {
  pred_db[year == res$year & id %in% res$ids, consolidated := res$preds]
}

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**  
- `data.table` for fast filtering and assignment.  
- Chunked prediction to avoid memory overload.  
- Parallelization to utilize multiple cores.  
- No retraining; models are reused as requested.  

This approach will significantly reduce runtime and memory usage on a normal machine.