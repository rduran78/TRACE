 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- `test_set` is created repeatedly with `dplyr::filter()`, which is slow for large data.
- Predictions are stored in `pred_db` via repeated indexing, which is inefficient.
- Entire dataset is kept in memory.

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment.
- Process data in **chunks** to reduce memory load.
- Avoid copying large objects repeatedly.
- Use `fwrite()` for efficient output.
- Keep models in memory but do not retrain.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Ensure year is character for matching
joined_data[, year := as.character(year)]
pred_db[, year := as.character(year)]

# Initialize consolidated column
pred_db[, consolidated := NA_real_]

# Process by year in chunks
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    
    # Filter rows for this year
    test_set <- joined_data[year == yr]
    
    # Predict in chunks to save memory
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions back efficiently
    pred_db[year == yr, consolidated := preds]
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically speeds up filtering and assignment.
- Chunked prediction prevents memory overload.
- Avoids repeated object copies.
- Preserves trained models and uses them as-is.

This approach should make the process feasible on a normal machine without retraining.