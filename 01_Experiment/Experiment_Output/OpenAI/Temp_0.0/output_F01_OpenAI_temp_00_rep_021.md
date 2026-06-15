 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- `joined_data` is filtered repeatedly, creating large intermediate objects.
- Predictions are done sequentially without chunking, causing high RAM usage.
- `pred_db` is updated in place, which is inefficient for large vectors.

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment.
- Process predictions in chunks to reduce memory load.
- Avoid copying large objects repeatedly.
- Pre-allocate prediction column efficiently.
- Use `fwrite()` for faster output.

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Pre-allocate prediction column
pred_db[, consolidated := NA_real_]

# Define chunk size (adjust based on RAM)
chunk_size <- 50000

for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset once
    test_set <- joined_data[year == yr]
    idx <- which(pred_db$year == yr)
    
    # Chunked prediction
    n <- nrow(test_set)
    for (i in seq(1, n, by = chunk_size)) {
      j <- min(i + chunk_size - 1, n)
      pred_chunk <- predict(rf_model, newdata = test_set[i:j])
      pred_db$consolidated[idx[i:j]] <- pred_chunk
    }
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Fast write
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` minimizes memory copies.
- Chunking prevents loading the entire year's data into `predict()` at once.
- Pre-allocation avoids repeated vector resizing.
- `fwrite()` is much faster than `write.csv()` for large files.

This approach should make the process feasible on a normal machine without retraining models.