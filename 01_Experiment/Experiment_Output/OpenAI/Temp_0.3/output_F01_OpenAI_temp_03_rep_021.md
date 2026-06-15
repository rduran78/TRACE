 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets combined with `dplyr` filtering and `predict()` calls on full data frames. Each iteration copies large objects in memory, and `predict()` on large `data.frame`s is slow.  

**Optimization Strategy:**  
1. **Avoid repeated filtering:** Split `joined_data` by year once instead of filtering in each loop.  
2. **Use `data.table` for memory efficiency:** Convert large data frames to `data.table`.  
3. **Vectorized prediction:** Apply predictions in chunks to avoid memory overflow.  
4. **Parallelization (optional):** Use `parallel::mclapply` or `future.apply` if multiple cores are available.  
5. **Write output incrementally:** Avoid holding the entire result in memory by writing in chunks.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, by = "year")

# Prepare output column
pred_db[, consolidated := NA_real_]

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

# Loop through years efficiently
for (yr in names(data_by_year)) {
  cat("Predicting for year", yr, "\n")
  
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    test_set <- data_by_year[[yr]]
    
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign predictions
    pred_db[year == as.integer(yr), consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- **`data.table`** reduces memory overhead and speeds up filtering/assignment.  
- **Chunked prediction** prevents RAM exhaustion.  
- **Single split upfront** avoids repeated costly filtering.  
- **`fwrite`** is faster and more memory-efficient than `write.csv`.  

This approach should make the process feasible on a standard machine without retraining models.