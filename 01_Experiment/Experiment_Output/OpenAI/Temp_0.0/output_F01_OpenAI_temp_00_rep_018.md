 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- `joined_data` is filtered repeatedly, creating large intermediate objects.
- Predictions are done sequentially without chunking, causing high RAM usage.
- `pred_db` is updated in place, which is inefficient for large vectors.

---

**Optimization Strategy:**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.
2. **Chunk predictions** for each year to avoid loading the entire year's data into memory at once.
3. **Avoid repeated copying** by pre-allocating and updating in place.
4. **Leverage `parallel` prediction** if multiple cores are available.
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

# Process each year
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter data for this year
    test_set <- joined_data[year == yr]
    
    # Predict in chunks
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign predictions back efficiently
    pred_db[year == yr, consolidated := preds]
    
    # Optional: write intermediate results to disk
    fwrite(pred_db[year == yr], paste0("predictions_year_", yr, ".csv"))
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Save final result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**
- `data.table` drastically reduces filtering and assignment overhead.
- Chunked prediction prevents memory overload.
- Optional intermediate writes safeguard against crashes.
- Parallelization can be added by splitting `years` across cores using `mclapply()` if needed.

This approach makes the process feasible on a normal machine without retraining models.