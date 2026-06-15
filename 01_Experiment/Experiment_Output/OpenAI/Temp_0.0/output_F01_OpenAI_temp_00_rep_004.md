 **Diagnosis**  
The bottleneck is the loop predicting year by year on large data frames using `dplyr` and `predict()` on full subsets. This causes high memory usage and slow performance because:  
- `joined_data %>% filter(year == year)` repeatedly copies large data.  
- Predictions are done on large chunks without batching.  
- `pred_db$consolidated[...] <- ...` repeatedly modifies a large object in memory.  

**Optimization Strategy**  
- Use `data.table` for fast filtering and assignment (avoids repeated copies).  
- Process predictions in **chunks** to reduce memory load.  
- Avoid growing objects inside the loop; preallocate and update by reference.  
- Use `predict(..., newdata, type="response")` in batches for large datasets.  

**Optimized R Code**  
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Preallocate consolidated column if not present
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Define chunk size (adjust based on memory)
chunk_size <- 50000

years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset rows for this year
    idx <- which(pred_db$year == yr)
    test_set <- joined_data[year == yr]
    
    # Process in chunks
    n <- nrow(test_set)
    for (start in seq(1, n, by = chunk_size)) {
      end <- min(start + chunk_size - 1, n)
      preds <- predict(rf_model, newdata = test_set[start:end])
      pred_db$consolidated[idx[start:end]] <- preds
    }
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` avoids expensive copies and supports fast filtering and assignment.  
- Chunking prevents loading the entire year's data into memory for prediction.  
- Preallocation and in-place updates reduce overhead.  

This approach should make the process feasible on a normal machine without retraining models.