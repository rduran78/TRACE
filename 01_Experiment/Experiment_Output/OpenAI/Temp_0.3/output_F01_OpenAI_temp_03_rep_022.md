 **Diagnosis**  
The main bottleneck is the loop performing predictions year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:  
- `joined_data` is large and repeatedly filtered.  
- Predictions are done sequentially without leveraging vectorization or chunking.  
- `pred_db` is updated in place, which can be slow for large objects.  

**Optimization Strategy**  
- Use `data.table` for efficient filtering and assignment.  
- Predict in **chunks** to avoid loading the entire dataset into memory at once.  
- Avoid repeated subsetting of large data frames inside the loop.  
- Pre-split data by year and process sequentially, writing intermediate results to disk if needed.  

**Optimized R Code**  
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure rf_models_per_year is loaded
# rf_models_per_year <- ... (already loaded from RData)

# Create an output vector
pred_db[, consolidated := NA_real_]

# Define chunk size (adjust based on memory)
chunk_size <- 50000

for (year in unique(pred_db$year)) {
  cat("Predicting for year", year, "\n")
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Subset once
    test_set <- joined_data[year == !!year]
    
    # Process in chunks
    n <- nrow(test_set)
    if (n > 0) {
      preds <- numeric(n)
      idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
      
      for (i in seq_along(idx)) {
        preds[idx[[i]]] <- predict(rf_model, newdata = test_set[idx[[i]], ])
      }
      
      # Assign predictions back efficiently
      pred_db[year == !!year, consolidated := preds]
    }
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why This Works**  
- `data.table` speeds up filtering and assignment.  
- Chunking prevents memory overload during `predict()`.  
- Avoids repeated expensive operations by subsetting once per year.  

This approach should make the process feasible on a normal machine while preserving the trained models.