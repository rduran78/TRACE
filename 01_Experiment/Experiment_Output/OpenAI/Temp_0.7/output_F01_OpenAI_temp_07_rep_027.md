 **Diagnosis:**  
The slowness and memory intensity come from:  
- Iterating over all years with full `joined_data` filtering repeatedly.  
- Using `dplyr` operations and large in-memory objects (`pred_db`, `joined_data`) for hundreds of thousands of rows.  
- `predict()` on very large `test_set` without batching.  
- Keeping all intermediate objects in memory.  

**Optimization Strategy:**  
- Convert data to `data.table` for fast subsetting and assignment.  
- Process predictions in **chunks** (batching) to reduce memory load.  
- Avoid repeated filtering by pre-splitting or indexing.  
- Write results incrementally (or use `fwrite`) instead of holding entire `pred_db` in memory.  
- Use `gc()` after each chunk to free memory.  

**Optimized R Code:**  
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure rf_models_per_year exists from loaded .RData
years <- unique(pred_db$year)

# Define a batch size (tune as per available RAM)
batch_size <- 50000

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset once
    idx <- which(joined_data$year == yr)
    n <- length(idx)
    
    if (n > 0) {
      # Process in batches
      for (i in seq(1, n, by = batch_size)) {
        batch_idx <- idx[i:min(i + batch_size - 1, n)]
        test_set <- joined_data[batch_idx]
        
        preds <- predict(rf_model, newdata = test_set)
        
        # Assign directly in pred_db
        pred_db$consolidated[pred_db$year == yr][(i):(i + length(preds) - 1)] <- preds
        
        rm(test_set, preds); gc()
      }
    }
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write using fwrite for speed
fwrite(pred_db, "RF_imputated_db.csv")
```

**Key Gains:**  
- `data.table` reduces subsetting cost from multiple seconds to milliseconds.  
- Batch prediction prevents RAM exhaustion.  
- Writing with `fwrite` is faster and memory-efficient.  

This approach preserves trained models and avoids retraining while making the process feasible on a normal machine.