 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` on full data frames in memory. This approach is memory-intensive and slow because:
- `joined_data` is large and repeatedly filtered.
- Predictions are done on the entire subset at once.
- No chunking or parallelization is used.
- `pred_db` is updated in a vectorized but inefficient way for large data.

---

**Optimization Strategy:**  
- Use **data.table** for fast filtering and assignment.
- Predict in **chunks** to reduce memory load.
- Use **parallel processing** with `parallel::mclapply` or `future.apply`.
- Avoid copying large objects repeatedly.
- Keep models as is (no retraining).

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
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  preds
}

# Parallel prediction by year
years <- unique(pred_db$year)

mclapply(years, function(year) {
  cat("Predicting for year", year, "\n")
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict_in_chunks(rf_model, test_set)
    pred_db[year == !!year, consolidated := preds]
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}, mc.cores = n_cores)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**
- `data.table` for efficient filtering and assignment.
- Chunked prediction to avoid memory overload.
- Parallel processing across years for speed.
- Preserves trained models and avoids retraining.  

This approach will significantly reduce memory usage and runtime on a personal computer.