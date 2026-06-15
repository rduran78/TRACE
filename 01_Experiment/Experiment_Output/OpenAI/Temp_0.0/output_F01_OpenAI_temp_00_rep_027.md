 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- `test_set` is created repeatedly with `dplyr::filter()`, which is slow for large data.
- Predictions are stored in `pred_db` via repeated indexing, which is inefficient.
- Everything runs in a single thread.

---

**Optimization Strategy:**  
1. **Use `data.table` for filtering and assignment** (fast and memory-efficient).
2. **Chunk predictions** for large datasets to avoid memory overload.
3. **Parallelize predictions** using `parallel` or `future.apply`.
4. **Avoid copying large objects repeatedly** by pre-allocating and updating in place.

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Detect cores for parallelization
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

# Parallel prediction by year
years <- unique(pred_db$year)

cl <- makeCluster(n_cores)
clusterExport(cl, c("joined_data", "pred_db", "rf_models_per_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

parLapply(cl, years, function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict_in_chunks(rf_model, test_set)
    pred_db[year == !!year, consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", year, "\n"))
  }
})

stopCluster(cl)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Gains:**
- `data.table` drastically reduces filtering and assignment time.
- Chunking prevents memory overload during `predict()`.
- Parallelization uses multiple cores for speed.
- Preserves trained models and avoids retraining.  

This approach should make the process feasible on a normal machine.