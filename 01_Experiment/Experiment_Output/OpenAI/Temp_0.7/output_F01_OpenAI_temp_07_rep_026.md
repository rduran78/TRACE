 **Diagnosis:**  
The current approach processes predictions year by year in-memory on large datasets, creating performance and memory bottlenecks because:  
- `joined_data` is filtered for each year repeatedly (inefficient subsetting).  
- Predictions for entire yearly subsets remain in memory.  
- No batching or disk-based processing for large data frames.  

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment.  
- Predict in **chunks (batch processing)** to avoid memory overload.  
- Avoid creating large intermediate objects; write results incrementally to disk.  
- Preserve existing `rf_models_per_year` without retraining.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Prepare output file
output_file <- "RF_imputated_db.csv"
fwrite(pred_db[0], output_file)  # write header

# Chunk size for prediction (adjust based on memory)
chunk_size <- 50000  

for (year in unique(pred_db$year)) {
  cat(paste0("Predicting for year ", year, "\n"))
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Subset once, avoid repeated filtering
    test_set <- joined_data[year == !!year]
    
    # Predict in chunks
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Update pred_db efficiently
    pred_db[year == !!year, consolidated := preds]
    
    # Optionally write incrementally to disk
    fwrite(pred_db[year == !!year], output_file, append = TRUE)
    
  } else {
    cat(paste0("Warning: No model found for year ", year, "\n"))
  }
}

cat("Prediction completed. Results saved to:", output_file, "\n")
```

---

**Why this works:**  
- `data.table` drastically speeds up filtering and assignment.  
- Chunked predictions prevent exhausting RAM on large datasets.  
- Incremental writing avoids holding the entire result in memory.  

**Extra tip:** If disk space allows, use `fst` or `arrow` for faster I/O instead of CSV.  

This approach makes the process feasible on a standard machine without retraining models.