 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets, combined with `dplyr` filtering and base assignment. This is memory-intensive and slow because:  
- `joined_data %>% filter(year == year)` repeatedly scans a large dataset.  
- Predictions are done sequentially without batching or parallelization.  
- `pred_db` updates are inefficient for large objects.  

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment.  
- Avoid copying large data frames repeatedly.  
- Parallelize predictions using `parallel` or `future.apply`.  
- Process in chunks if memory is limited.  
- Preserve trained models by loading them as before.  

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Detect available cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Function to predict for one year
predict_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]  # fast subset
    preds <- predict(rf_model, newdata = test_set)
    list(year = year, preds = preds)
  } else {
    message("Warning: No model found for year ", year)
    NULL
  }
}

# Parallel prediction
years <- unique(pred_db$year)
results <- mclapply(years, predict_year, mc.cores = n_cores)

# Combine results efficiently
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` reduces memory overhead and speeds up filtering and assignment.  
- `mclapply` leverages multiple cores for parallel prediction.  
- In-place updates avoid creating large intermediate copies.  
- Preserves trained models without retraining.  

This approach should significantly reduce runtime and memory usage on a normal machine.