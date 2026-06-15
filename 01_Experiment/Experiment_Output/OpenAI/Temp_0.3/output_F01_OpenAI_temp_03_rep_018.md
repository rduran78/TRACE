 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large `data.frame`s with `dplyr` filtering and `predict()` on full subsets. This causes repeated memory allocations and slow operations. Since models are already trained, the main goal is to optimize prediction by:  
- Avoiding repeated `dplyr` operations inside the loop.  
- Using `data.table` for fast filtering and assignment.  
- Predicting in chunks to reduce memory pressure.  
- Optionally leveraging parallelization if multiple cores are available.  

---

**Optimization Strategy:**  
1. Convert `pred_db` and `joined_data` to `data.table` for efficient subsetting and assignment.  
2. Pre-split `joined_data` by year to avoid repeated filtering.  
3. Use chunked prediction for very large subsets.  
4. Optionally enable parallel prediction using `parallel::mclapply` or `future.apply`.  

---

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# Pre-split joined_data by year
joined_split <- split(joined_data, by = "year")

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
years <- unique(pred_db$year)
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("rf_models_per_year", "joined_split", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, years, function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_split[[as.character(year)]]
    preds <- predict_in_chunks(rf_model, test_set)
    list(year = year, preds = preds)
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

**Why this works:**  
- `data.table` drastically reduces filtering and assignment overhead.  
- Chunked prediction prevents memory overload.  
- Parallelization utilizes multiple cores for speed.  
- Preserves trained models without retraining.  

This approach should make the process feasible on a normal machine while handling large datasets efficiently.