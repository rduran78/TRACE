 **Diagnosis:**  
The main bottlenecks are:  
- Iterating through years in R with `for` + `predict()` on large data frames (hundreds of thousands of rows and many predictors) is slow.  
- Using `dplyr` on such large data can be memory-intensive.  
- `randomForest::predict()` is single-threaded by default, leading to long runtimes.  

---

**Optimization Strategy:**  
1. Use **`data.table`** instead of `dplyr` for filtering and assignment—faster and memory-efficient.  
2. Enable **parallel prediction** using `foreach` and `parallel` or `future.apply`.  
3. Avoid copying data repeatedly; pre-split test sets by year.  
4. Process in chunks if memory is tight.  

---

**Optimized R Code:**  
```r
library(data.table)
library(parallel)
library(foreach)
library(doParallel)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

years <- unique(pred_db$year)

# Setup parallel backend
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

pred_db[, consolidated := NA_real_]

# Parallel prediction over years
foreach(yr = years, .packages = c("data.table", "randomForest")) %dopar% {
  message("Predicting for year ", yr)
  
  # Filter rows for this year
  test_set <- joined_data[year == yr]
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    preds <- predict(rf_model, newdata = test_set)
    
    # Assign predictions efficiently
    pred_idx <- which(pred_db$year == yr)
    pred_db$consolidated[pred_idx] <- preds
  } else {
    message("Warning: No model found for year ", yr)
  }
}

stopCluster(cl)

fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**  
- Uses `data.table` for fast filtering and assignment.  
- Parallelizes predictions across all available cores.  
- Avoids retraining models, preserves original trained models.  
- Handles large data without unnecessary copies.  

If still memory-constrained, add **chunked prediction** within each year:  
```r
chunk_size <- 50000
for (start in seq(1, nrow(test_set), by = chunk_size)) {
  end <- min(start + chunk_size - 1, nrow(test_set))
  preds[start:end] <- predict(rf_model, newdata = test_set[start:end])
}
```

This ensures scalability on a normal machine.