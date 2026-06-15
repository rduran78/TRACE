 **Diagnosis:**  
The current bottlenecks are:  
1. Iterating over years in R using a `for` loop with repeated `filter()` calls on large data frames (slow and memory-heavy).  
2. Predicting on large datasets in one go without chunking (high RAM usage).  
3. Using `dplyr` inside loops instead of more efficient methods like `data.table`.  

**Optimization Strategy:**  
- Use `data.table` for filtering and assignment (faster, memory-efficient).  
- Process predictions in **chunks** (batching rows) to reduce memory load.  
- Avoid repeatedly subsetting `pred_db` by switching to keyed `data.table`.  
- Keep models in memory as they are (no retraining).  

---

### **Optimized R Code**
```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Key by year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Define batch size for chunked predictions
batch_size <- 50000  # Adjust based on available RAM

years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Get test set for this year
    test_set <- joined_data[year == yr]
    n <- nrow(test_set)
    
    if (n > 0) {
      pred_vals <- numeric(n)
      
      # Process in chunks
      for (i in seq(1, n, by = batch_size)) {
        idx <- i:min(i + batch_size - 1, n)
        pred_vals[idx] <- predict(rf_model, newdata = test_set[idx])
      }
      
      # Assign predictions efficiently
      pred_db[year == yr, consolidated := pred_vals]
    }
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why This Works:**  
- `data.table` reduces overhead of filtering and assignment.  
- Chunked prediction prevents exhausting RAM on large datasets.  
- Maintains the original trained models and workflow.  

This approach makes the script feasible on a normal machine without changing model integrity.