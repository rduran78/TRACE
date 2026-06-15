 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- `joined_data` is filtered repeatedly, creating large intermediate objects.
- Predictions are done sequentially without chunking, so the entire year's data is loaded into memory at once.
- `pred_db` is updated in a non-vectorized way.

**Optimization Strategy:**  
- Use `data.table` for efficient filtering and assignment.
- Process predictions in **chunks** to avoid loading all rows at once.
- Avoid repeated `dplyr` operations inside the loop; pre-split data by year.
- Use `fwrite()` instead of `write.csv()` for speed.
- Keep models in memory but avoid retraining.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Pre-split joined_data by year to avoid repeated filtering
data_by_year <- split(joined_data, by = "year")

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

# Loop through years
for (year in names(data_by_year)) {
  cat("Predicting for year", year, "\n")
  
  if (year %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[year]]
    test_set <- data_by_year[[year]]
    
    # Predict in chunks
    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000)
    
    # Assign predictions efficiently
    pred_db[year == as.integer(year), consolidated := preds]
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically reduces memory overhead and speeds up filtering/assignment.
- Chunked prediction prevents memory overload for large datasets.
- Pre-splitting avoids repeated filtering inside the loop.
- `fwrite()` is much faster than `write.csv()` for large files.

This approach should make the process feasible on a normal machine without retraining models.