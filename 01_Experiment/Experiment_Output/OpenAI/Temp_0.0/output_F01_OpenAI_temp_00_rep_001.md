 **Diagnosis**  
The bottleneck is the loop predicting year by year on large data frames using `dplyr` and `predict()` on full subsets in memory. This causes high memory usage and slow performance because:  
- `joined_data %>% filter(year == year)` repeatedly copies large data.  
- Predictions are done on large chunks without batching.  
- `pred_db$consolidated[...] <- ...` repeatedly modifies a large object in memory.  

**Optimization Strategy**  
- Use `data.table` for fast filtering and assignment (avoids repeated copies).  
- Process data in chunks (batch prediction) to reduce memory load.  
- Avoid growing objects inside the loop; preallocate and update by reference.  
- Use `fwrite()` for faster output.  

**Optimized R Code**  
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Preallocate consolidated column if not present
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Get unique years
years <- unique(pred_db$year)

# Loop through years
for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter rows for this year
    test_idx <- which(joined_data$year == yr)
    test_set <- joined_data[test_idx]
    
    # Predict in chunks to save memory
    chunk_size <- 50000
    preds <- numeric(length(test_idx))
    
    for (i in seq(1, length(test_idx), by = chunk_size)) {
      idx_range <- i:min(i + chunk_size - 1, length(test_idx))
      preds[idx_range] <- predict(rf_model, newdata = test_set[idx_range])
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
- `data.table` avoids expensive copies and supports in-place updates.  
- Chunked prediction prevents memory overload.  
- `fwrite()` is much faster than `write.csv()`.  

This approach preserves the trained models and makes the process feasible on a normal machine.