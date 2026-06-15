 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- `joined_data` is filtered repeatedly, creating large intermediate objects.
- Predictions are done sequentially without chunking, so the entire year's data is loaded into memory at once.
- `pred_db` is updated in place, which can be slow for large data frames.

---

**Optimization Strategy:**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.
2. **Chunk predictions** for each year to avoid loading all rows into memory at once.
3. **Avoid repeated copying** by pre-allocating and updating in place.
4. **Parallelize predictions** if possible using `parallel::mclapply` or `future.apply`.
5. **Write intermediate results to disk** to prevent memory overflow.

---

**Optimized R Code:**
```r
library(data.table)
library(randomForest)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Pre-allocate consolidated column
pred_db[, consolidated := NA_real_]

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

# Parallel processing setup
n_cores <- max(1, detectCores() - 1)

# Process each year
years <- unique(pred_db$year)

mclapply(years, function(yr) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter data for this year
    test_set <- joined_data[year == yr]
    
    # Predict in chunks
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign predictions back
    pred_db[year == yr, consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}, mc.cores = n_cores)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically reduces filtering and assignment overhead.
- Chunking prevents memory overload when predicting on large datasets.
- Parallelization speeds up processing across years.
- `fwrite` is faster and more memory-efficient than `write.csv`.

This approach preserves the trained models and makes the process feasible on a normal machine.