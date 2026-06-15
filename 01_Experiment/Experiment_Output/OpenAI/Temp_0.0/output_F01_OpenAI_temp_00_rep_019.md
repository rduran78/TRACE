 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated indexing. This approach is memory-inefficient and slow because:  
- `filter()` and subsetting inside the loop repeatedly copy large data frames.  
- `predict()` is called on large chunks without batching, which can exhaust memory.  
- `pred_db$consolidated[...] <- ...` repeatedly modifies a large object, causing overhead.  

**Optimization Strategy:**  
- Use `data.table` for fast filtering and in-place updates.  
- Process predictions in chunks (batching) to reduce memory load.  
- Avoid repeated copying by pre-sorting and joining results efficiently.  
- Keep models in memory but do not retrain.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure both have a common key for fast filtering
setkey(pred_db, year)
setkey(joined_data, year)

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

# Apply predictions by year
for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Get test set for this year
    test_set <- joined_data[year == yr]
    
    if (nrow(test_set) > 0) {
      preds <- predict_in_chunks(rf_model, test_set)
      
      # Update pred_db in place
      pred_db[year == yr, consolidated := preds]
    }
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` avoids repeated copying and speeds up filtering and assignment.  
- Chunked prediction prevents memory overload when handling large datasets.  
- In-place updates (`:=`) are highly efficient compared to repeated vector replacement.  

This approach should make the process feasible on a normal machine without retraining models.