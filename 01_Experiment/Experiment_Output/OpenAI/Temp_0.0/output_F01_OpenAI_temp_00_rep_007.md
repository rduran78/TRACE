 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated subsetting. This approach is memory-inefficient and slow because:  
- `filter()` and `%>%` create copies of large data frames repeatedly.  
- Predictions are done year by year instead of in a vectorized or chunked manner.  
- `predict()` on large data frames can exhaust memory if done in one go.  

**Optimization Strategy:**  
1. **Switch to `data.table` for fast filtering and assignment** (avoids repeated copies).  
2. **Chunk predictions** for large datasets to prevent memory overload.  
3. **Avoid unnecessary object duplication** by pre-allocating and updating in place.  
4. **Parallelize predictions** if multiple cores are available.  

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Pre-allocate consolidated column if not present
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Detect available cores for parallel processing
n_cores <- max(1, detectCores() - 1)

# Function to predict in chunks
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = newdata[idx[[i]], ])
  }
  preds
}

# Parallel loop over years
mclapply(unique(pred_db$year), function(year) {
  cat("Predicting for year", year, "\n")
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Subset test set efficiently
    test_set <- joined_data[year == !!year]
    
    # Predict in chunks
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign predictions back
    pred_db[year == !!year, consolidated := preds]
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}, mc.cores = n_cores)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` drastically reduces memory overhead and speeds up filtering/assignment.  
- Chunked prediction prevents memory crashes on large datasets.  
- `mclapply` uses multiple cores for parallel year-level processing.  
- Preserves trained models and avoids retraining.  

This approach should make the process feasible on a normal machine while handling large datasets efficiently.