 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` on full data frames in memory. This causes high memory usage and slow performance because:
- `joined_data` is large and repeatedly filtered.
- Predictions are done on the entire subset at once.
- No chunking or parallelization is used.

---

**Optimization Strategy:**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.
2. **Chunk predictions** for each year to avoid loading huge subsets into memory at once.
3. **Parallelize predictions** using `parallel` or `future.apply` if multiple cores are available.
4. **Avoid copying large objects** repeatedly; work with references.
5. **Write intermediate results to disk** to prevent memory overflow.

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Detect available cores for parallelization
n_cores <- max(1, detectCores() - 1)

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

# Parallel processing by year
years <- unique(pred_db$year)

cl <- makeCluster(n_cores)
clusterExport(cl, c("joined_data", "pred_db", "rf_models_per_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

parLapply(cl, years, function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict_in_chunks(model, test_set)
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

**Key Improvements:**
- `data.table` for fast filtering and assignment.
- Chunked predictions to reduce memory load.
- Parallelization across years for speed.
- Avoids retraining models and preserves existing Random Forest objects.

This approach makes the process feasible on a normal machine while handling large datasets efficiently.