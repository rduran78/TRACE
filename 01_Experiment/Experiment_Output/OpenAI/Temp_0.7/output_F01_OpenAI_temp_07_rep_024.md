 **Diagnosis:**  
The bottleneck lies in processing large datasets in memory and predicting year-by-year on full data frames. `predict()` on large `data.frame`s can be memory intensive. The loop loads entire subsets into RAM, and writing the full `pred_db` only at the end compounds memory pressure.

---

### **Optimization Strategy**
1. **Use memory-efficient structures:** Convert large data frames to `data.table` for fast filtering and assignment.
2. **Chunk processing:** Predict in batches to reduce RAM usage.
3. **Avoid repeated filtering:** Pre-split indices or use `split()` for efficient access.
4. **Write incrementally:** Append predictions to disk progressively to avoid holding all results in memory.
5. **Leverage parallel prediction (optional):** Use `parallel::mclapply` or `future.apply` if multiple cores are available.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Prepare output file
out_file <- "RF_imputated_db.csv"
fwrite(pred_db[0], out_file)  # write header only

# Iterate by year efficiently
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset using data.table fast filter
    test_idx <- pred_db$year == yr
    test_set <- joined_data[year == yr]
    
    # Predict in chunks to save memory
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      j <- min(i + chunk_size - 1, n)
      preds[i:j] <- predict(rf_model, newdata = test_set[i:j])
    }
    
    # Assign predictions without copying entire table
    pred_db[test_idx, consolidated := preds]
    
    # Write incrementally to disk
    fwrite(pred_db[test_idx], out_file, append = TRUE)
    
    # Remove temporary objects
    rm(test_set, preds)
    gc()
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

cat("Prediction complete. Results written to", out_file, "\n")
```

---

### **Why this works**
- **`data.table`** minimizes memory copies and speeds up filtering/assignment.
- **Chunked `predict()`** avoids loading the entire year's data into memory at once.
- **Incremental writing** prevents holding the entire augmented dataset in memory.
- Preserves trained Random Forest models and avoids retraining.

This approach should make processing feasible on a personal computer with large datasets.