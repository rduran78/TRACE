 **Diagnosis:**  
The bottleneck is the loop applying predictions year by year on large datasets in memory. `predict()` on Random Forest models for hundreds of thousands of rows per year is computationally heavy, and filtering `joined_data` repeatedly adds overhead. Everything runs sequentially, and the entire `pred_db` stays in memory.

---

**Optimization Strategy:**  
- Use **data.table** for fast filtering and assignment instead of `dplyr`.
- Avoid repeatedly copying data; pre-split data by year once.
- Apply **parallel processing** for predictions using `parallel` or `future.apply`.
- Write intermediate results to disk or use `fwrite()` for efficiency.
- Keep models as-is (no retraining).

---

**Optimized R Code:**

```r
library(data.table)
library(parallel)

# Convert joined_data and pred_db to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Get unique years
years <- unique(pred_db$year)

# Detect cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Split joined_data by year once
split_data <- split(joined_data, joined_data$year)

# Parallel prediction
cl <- makeCluster(n_cores)
clusterExport(cl, c("rf_models_per_year", "split_data", "pred_db"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, years, function(yr) {
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    test_set <- split_data[[as.character(yr)]]
    preds <- predict(rf_model, newdata = test_set)
    data.table(year = yr, consolidated = preds, id = test_set$id) # assuming 'id' exists
  } else {
    NULL
  }
})
stopCluster(cl)

# Combine results
results_dt <- rbindlist(results, use.names = TRUE, fill = TRUE)

# Merge predictions back into pred_db
pred_db[results_dt, consolidated := i.consolidated, on = .(year, id)]

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Gains:**  
- Eliminates repeated filtering with `dplyr`.
- Uses parallel cores for prediction.
- Reduces memory overhead by splitting data once and merging efficiently.
- Uses `fwrite()` for fast output.  

This approach should make the process feasible on a standard machine.