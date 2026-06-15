 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated subsetting. This approach is memory-inefficient and slow because:  
- `filter()` and `%>%` create copies of large data frames repeatedly.  
- Predictions are done year by year instead of in a vectorized or chunked manner.  
- `predict()` on large data frames can be memory-heavy if done in one go.  

**Optimization Strategy:**  
1. **Convert to `data.table`** for fast filtering and assignment without copying.  
2. **Chunk predictions** for large datasets to avoid memory overload.  
3. **Avoid repeated joins or copies**; work on the original `pred_db` directly.  
4. **Parallelize predictions** if multiple cores are available.  

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure keys for fast filtering
setkey(pred_db, year)
setkey(joined_data, year)

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
clusterExport(cl, c("rf_models_per_year", "joined_data", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

parLapply(cl, years, function(yr) {
  if (as.character(yr) %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[as.character(yr)]]
    test_set <- joined_data[year == yr]
    preds <- predict_in_chunks(model, test_set)
    list(year = yr, preds = preds)
  } else {
    NULL
  }
}) -> results

stopCluster(cl)

# Assign predictions back efficiently
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` avoids repeated copies and speeds up filtering/assignment.  
- Chunking prevents memory overload during `predict()`.  
- Parallelization uses multiple cores for different years.  
- Preserves trained models and avoids retraining.  

This approach should make the process feasible on a normal machine.