 **Diagnosis:**  
The bottleneck is the loop over years combined with `predict()` on large datasets. Each iteration processes hundreds of thousands of rows in memory, which is costly. Using `dplyr` and base R for filtering and assignment also adds overhead. Predictions can be vectorized or processed in chunks to reduce memory pressure.

---

**Optimization Strategy:**  
1. **Chunked prediction:** Split large `test_set` into manageable chunks to avoid memory overload.  
2. **Use `data.table`:** Efficient filtering and assignment for large datasets.  
3. **Avoid repeated copying:** Preallocate prediction column and update by reference.  
4. **Parallelization (optional):** Use `parallel::mclapply` or `future.apply` for multi-core prediction.  

---

**Optimized R Code:**
```r
library(data.table)
library(randomForest)

# Convert pred_db and joined_data to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Preallocate prediction column
pred_db[, consolidated := NA_real_]

# Define chunk size (adjust based on available RAM)
chunk_size <- 50000

for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter rows for this year
    test_set <- joined_data[year == yr]
    
    # Process in chunks
    n <- nrow(test_set)
    idx_seq <- seq(1, n, by = chunk_size)
    
    preds <- vector("numeric", n)
    
    for (i in idx_seq) {
      end_i <- min(i + chunk_size - 1, n)
      preds[i:end_i] <- predict(rf_model, newdata = test_set[i:end_i])
    }
    
    # Assign predictions by reference
    pred_db[year == yr, consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` minimizes memory copies and speeds up filtering/assignment.  
- Chunking prevents loading all rows into `predict()` at once.  
- Preallocation avoids repeated growing of vectors.  
- `fwrite` is faster than `write.csv`.  

This approach should significantly reduce memory usage and improve runtime on a personal computer without retraining models.