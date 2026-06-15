 **Diagnosis:**  
The bottleneck is the loop-based prediction on very large data (`hundreds of thousands of rows per year`) combined with `predict()` on full data frames. This approach is memory-intensive and slow because:  
- Entire `test_set` is loaded in memory for each year.  
- Predictions are done sequentially without batching or parallelization.  
- `pred_db` is updated in a non-vectorized way.  

**Optimization Strategy:**  
1. **Avoid copying large data repeatedly**: Use `data.table` for efficient filtering and assignment.  
2. **Batch predictions**: Split large datasets into manageable chunks to reduce memory load.  
3. **Parallelize predictions**: Use `parallel::mclapply` or `future.apply` for multi-core processing.  
4. **Write results incrementally**: Avoid holding the entire output in memory if possible.  

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Detect available cores
n_cores <- max(1, detectCores() - 1)

# Function to predict in chunks
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  preds <- vector("numeric", n)
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = newdata[idx[[i]], ])
  }
  preds
}

# Parallel processing by year
years <- unique(pred_db$year)

mclapply(years, function(year) {
  cat("Predicting for year", year, "\n")
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Filter rows for this year
    test_set <- joined_data[year == !!year]
    
    # Predict in chunks
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign predictions back efficiently
    pred_db[year == !!year, consolidated := preds]
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}, mc.cores = n_cores)

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically reduces memory overhead for filtering and assignment.  
- Chunked prediction prevents memory overflow for large datasets.  
- `mclapply` leverages multiple cores for parallel year-level processing.  
- `fwrite` is faster than `write.csv` for large files.  

This approach preserves the trained models, avoids retraining, and makes the process feasible on a normal machine.