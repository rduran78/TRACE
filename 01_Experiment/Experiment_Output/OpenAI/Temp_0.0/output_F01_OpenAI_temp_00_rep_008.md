 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated indexing. This approach is memory-inefficient and slow because:  
- `filter()` and subsetting inside the loop repeatedly copy large data frames.  
- `predict()` is called on large chunks without batching or parallelization.  
- `pred_db$consolidated[...] <- ...` repeatedly modifies a large object, causing memory overhead.  

---

### **Optimization Strategy**
1. **Avoid repeated filtering:** Split `joined_data` by year once, not inside the loop.  
2. **Use `data.table` for efficient subsetting and assignment.**  
3. **Batch predictions or process in chunks to reduce memory spikes.**  
4. **Parallelize predictions using `parallel` or `future.apply`.**  
5. **Write results incrementally to disk instead of holding everything in memory.**  

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, joined_data$year)

# Detect available cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Function to predict for one year
predict_for_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- data_by_year[[as.character(year)]]
    
    # Predict in chunks to avoid memory overload
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
    
    return(data.table(year = year, pred = preds, id = test_set$id)) # assuming 'id' exists
  } else {
    warning(paste("No model found for year", year))
    return(NULL)
  }
}

# Run in parallel
years <- unique(pred_db$year)
results_list <- mclapply(years, predict_for_year, mc.cores = n_cores)

# Combine predictions
predictions <- rbindlist(results_list, use.names = TRUE, fill = TRUE)

# Merge back into pred_db efficiently
setkey(pred_db, year, id)
setkey(predictions, year, id)
pred_db[predictions, consolidated := i.pred]

# Write to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Why This Works**
- **`data.table`** drastically reduces memory copies and speeds up joins.  
- **Chunked prediction** prevents RAM exhaustion.  
- **Parallelization** uses multiple cores for faster execution.  
- **Incremental merging** avoids repeated large object modifications.  

This approach should make the process feasible on a normal machine without retraining models.