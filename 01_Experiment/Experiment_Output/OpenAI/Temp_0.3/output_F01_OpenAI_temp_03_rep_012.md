 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large data frames using `dplyr::filter` and `predict` sequentially. This causes repeated subsetting, high memory usage, and no parallelization. Since models are already trained, the main optimization should focus on:  
- Avoiding repeated filtering with `dplyr` (slow for large data).  
- Using `data.table` for efficient grouping and assignment.  
- Leveraging parallel prediction if multiple cores are available.  
- Optionally processing data in chunks to reduce memory footprint.  

---

### **Optimization Strategy**
1. Convert `joined_data` and `pred_db` to `data.table` for fast filtering and assignment.
2. Use `split` or `data.table` grouping to avoid repeated filtering.
3. Predict in parallel using `parallel::mclapply` or `future.apply`.
4. Write results incrementally or in chunks if memory is tight.

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Get unique years and available cores
years <- unique(pred_db$year)
n_cores <- max(1, detectCores() - 1)

# Parallel prediction by year
results <- mclapply(years, function(yr) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    test_set <- joined_data[year == yr]
    preds <- predict(rf_model, newdata = test_set)
    data.table(id = test_set$id, pred = preds)  # assuming 'id' is unique key
  } else {
    cat("Warning: No model found for year", yr, "\n")
    NULL
  }
}, mc.cores = n_cores)

# Combine predictions
results_dt <- rbindlist(results, use.names = TRUE, fill = TRUE)

# Merge predictions back to pred_db by id
setkey(pred_db, id)
setkey(results_dt, id)
pred_db[results_dt, consolidated := i.pred]

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Why This Works**
- `data.table` drastically reduces filtering and assignment overhead.
- Parallelization uses multiple cores for independent yearly predictions.
- Memory is managed by processing year-wise subsets without duplicating the entire dataset repeatedly.
- Preserves trained models and avoids retraining.

This approach should make the process significantly faster and feasible on a normal machine.