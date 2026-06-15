 **Diagnosis:**  
The main bottleneck is the loop that processes predictions year by year on large datasets (`hundreds of thousands of cells` + multiple predictors). Each iteration creates large filtered subsets and uses `predict()` repeatedly, which is memory and time intensive. Additionally, `dplyr`’s `filter()` in a loop is slower and memory-heavy compared to `data.table`.  

**Optimization Strategy:**  
- Avoid repeated `filter()` calls with `dplyr`; use `data.table` for efficient subsetting.  
- Precompute an index mapping years to row positions to avoid scanning data repeatedly.  
- Use `predict(..., newdata)` in chunks (batching) to reduce memory pressure.  
- Leverage `data.table` in-place updates instead of copying large data frames.  
- Keep models intact (as required).  

---

### **Optimized R Code**
```r
library(data.table)

# Convert to data.table for efficient operations
setDT(pred_db)
setDT(joined_data)

# Precompute row indices by year
year_index <- split(seq_len(nrow(joined_data)), joined_data$year)

# Ensure consolidated column exists
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Define a batch size for predictions to reduce memory pressure
batch_size <- 50000  # adjust based on available RAM

for (yr in names(year_index)) {
  cat("Predicting for year", yr, "\n")
  
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    
    idx <- year_index[[yr]]
    test_set <- joined_data[idx]
    
    # Predict in batches
    n <- nrow(test_set)
    preds <- numeric(n)
    for (i in seq(1, n, by = batch_size)) {
      j <- min(i + batch_size - 1, n)
      preds[i:j] <- predict(rf_model, newdata = test_set[i:j])
    }
    
    # Assign predictions back efficiently
    pred_db[year == as.integer(yr), consolidated := preds]
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` avoids copying large datasets in the loop.  
- Precomputed year indices eliminate repeated filtering overhead.  
- Batch prediction prevents memory overload from loading huge chunks at once.  
- In-place assignment (`:=`) is fast and memory efficient.  

This approach should significantly reduce runtime and memory usage while preserving existing trained models.