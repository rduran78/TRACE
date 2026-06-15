 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated subsetting. This approach is memory-inefficient and slow because:  
- `filter()` and `%>%` create copies of large data frames repeatedly.  
- Predictions are done year by year instead of in a vectorized or chunked manner.  
- `predict()` on large data frames can be memory-heavy if done in one go.  

**Optimization Strategy:**  
1. **Switch to `data.table` for fast filtering and assignment** (avoids repeated copies).  
2. **Chunk predictions** for large datasets to reduce memory load.  
3. **Avoid unnecessary joins or transformations inside the loop**—prepare data once.  
4. **Use `set()` for in-place assignment** in `data.table`.  
5. **Parallelize predictions** if possible (optional, using `parallel` or `future.apply`).  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure keys for fast filtering
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
  return(preds)
}

# Loop through years
for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset test set efficiently
    test_set <- joined_data[year == yr]
    
    # Predict in chunks
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign predictions in place
    pred_db[year == yr, consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` avoids repeated copies and speeds up filtering/assignment.  
- Chunking prevents memory overload during `predict()`.  
- In-place assignment (`:=`) is highly memory-efficient.  
- This approach preserves the trained models and does not retrain them.  

**Optional further optimization:**  
- Use `parallel::mclapply` or `future.apply` inside `predict_in_chunks` for multi-core predictions.  
- If predictors are sparse, consider converting to `matrix` for faster prediction.