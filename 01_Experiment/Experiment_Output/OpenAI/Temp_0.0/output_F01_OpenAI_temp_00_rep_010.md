 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated indexing. This approach is memory-inefficient and slow because:
- `filter()` and subsetting inside the loop repeatedly copy large data frames.
- Predictions are done year by year instead of in a vectorized or chunked manner.
- `pred_db$consolidated[...] <- ...` triggers repeated writes to a large object.

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment (avoids repeated copies).
- Process predictions in chunks to reduce memory load.
- Avoid creating intermediate large objects unnecessarily.
- Keep models as is (no retraining).
- Write results incrementally or after all predictions to avoid repeated I/O.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure 'year' is keyed for fast filtering
setkey(pred_db, year)
setkey(joined_data, year)

# Initialize consolidated column
pred_db[, consolidated := NA_real_]

# Loop through years efficiently
for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Get test set for this year
    test_set <- joined_data[J(yr)]
    
    # Predict in chunks to avoid memory overload
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions back efficiently
    pred_db[J(yr), consolidated := preds]
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output once
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` subsetting and assignment are in-place (no large copies).
- Chunked prediction prevents memory spikes.
- Single write at the end avoids repeated disk I/O.
- Preserves trained models and logic.

This approach should make the process feasible on a normal machine without retraining.