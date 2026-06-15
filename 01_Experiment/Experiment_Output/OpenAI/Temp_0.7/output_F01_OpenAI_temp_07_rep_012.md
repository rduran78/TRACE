 **Diagnosis:**  
The bottleneck is the loop performing `predict()` on very large subsets of data without memory optimization. This involves:  
- Multiple full-data copies (`dplyr::filter()` inside the loop).  
- Predictions applied year-by-year instead of using chunking or vectorization.  
- `pred_db` updates inside the loop, which is inefficient for large objects.  

**Optimization Strategy:**  
- Use `data.table` for efficient filtering and updates in place.  
- Predict in chunks to avoid memory overload.  
- Pre-allocate prediction column and avoid repeated copying.  
- Use `rf_model` predictions on a `data.table` subset without converting back to `data.frame`.  
- Optionally parallelize predictions with `parallel::mclapply` if cores are available.  

---

### **Optimized R Code**
```r
library(data.table)

# Convert data to data.table for efficient processing
setDT(pred_db)
setDT(joined_data)

# Pre-allocate prediction column
pred_db[, consolidated := NA_real_]

years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset test_set efficiently
    test_idx <- joined_data$year == yr
    test_set <- joined_data[test_idx]
    
    # Predict in chunks to reduce memory usage
    chunk_size <- 50000
    preds <- numeric(nrow(test_set))
    
    for (i in seq(1, nrow(test_set), by = chunk_size)) {
      j <- min(i + chunk_size - 1, nrow(test_set))
      preds[i:j] <- predict(rf_model, newdata = test_set[i:j])
    }
    
    # Assign predictions back using data.table indexing
    pred_db[year == yr, consolidated := preds]
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this is more efficient:**  
- `data.table` avoids repeated object copying.  
- Chunking prevents exhausting RAM during prediction.  
- Updates are done by reference (`:=`), reducing overhead.  
- `fwrite()` is faster than `write.csv`.  

This approach keeps the trained models intact, minimizes memory usage, and scales on a normal machine.