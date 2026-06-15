 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large data frames, which causes high memory usage and slow performance. The entire `joined_data` and `pred_db` are kept in memory, and `predict()` on large subsets is expensive.  

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment instead of `dplyr`.  
- Process predictions in **chunks** to avoid loading all rows into memory at once.  
- Avoid copying large objects repeatedly; work with references.  
- Write intermediate results to disk incrementally instead of holding everything in RAM.  

**Optimized R Code:**  
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure keys for fast filtering
setkey(pred_db, year)
setkey(joined_data, year)

# Open a connection for incremental writing
fwrite(pred_db[0], "RF_imputated_db.csv")  # write header only

for (year in unique(pred_db$year)) {
  cat("Predicting for year", year, "\n")
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Get test set for this year
    test_set <- joined_data[J(year)]
    
    # Predict in chunks to reduce memory load
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions
    pred_db[year == ..year, consolidated := preds]
    
    # Write incrementally to disk
    fwrite(pred_db[year == ..year], "RF_imputated_db.csv", append = TRUE)
    
    # Remove from memory
    rm(test_set, preds)
    gc()
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}

cat("Prediction complete. Results saved to RF_imputated_db.csv\n")
```

**Key Improvements:**  
- `data.table` for fast filtering and assignment.  
- Chunked prediction to avoid memory overload.  
- Incremental writing to CSV to keep RAM usage low.  

This approach preserves trained models, avoids retraining, and makes the process feasible on a personal computer.