 **Diagnosis**  
The main bottleneck is the `for` loop predicting year-by-year on large datasets. This causes:  
- Repeated subsetting (`filter`) and NA assignments on large data frames.  
- Lack of vectorization and memory-efficient data structures.  
- `predict()` on huge data in memory at once.  

**Optimization Strategy**  
- Use `data.table` for efficient filtering and column assignment (avoids data copies).  
- Split big predictions into manageable chunks (batch processing) to limit memory usage.  
- Use in-place assignment to avoid growing objects repeatedly.  
- Read only necessary columns in `joined_data` for prediction.  

**Optimized Code**  
```r
library(data.table)

# Convert to data.table for efficient handling
setDT(pred_db)
setDT(joined_data)

# Ensure keys for fast filtering
setkey(pred_db, year)
setkey(joined_data, year)

# Batch size for processing (adjust as needed)
batch_size <- 50000

for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset data for current year
    test_set <- joined_data[year == yr]
    
    # Split into chunks to reduce memory pressure
    n <- nrow(test_set)
    idx <- split(seq_len(n), ceiling(seq_len(n) / batch_size))
    
    predictions <- vector("numeric", n)
    
    for (i in seq_along(idx)) {
      chunk <- test_set[idx[[i]], ]
      predictions[idx[[i]]] <- predict(rf_model, newdata = chunk)
    }
    
    # Assign predictions directly using join
    pred_db[year == yr, consolidated := predictions]
    
  } else {
    cat("Warning: No model for year", yr, "\n")
  }
}

fwrite(pred_db, "RF_imputated_db.csv")  # Faster than write.csv
```

**Summary of Benefits**  
- Reduces memory overhead by chunked prediction.  
- Fast filtering and in-place modification via `data.table`.  
- Preserves existing trained models; no retraining needed.  
- Scales better on personal machines.