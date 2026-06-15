 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets (`joined_data`) and writing results into `pred_db`. This approach is memory-intensive because:
- `joined_data` and `pred_db` are large and repeatedly subset.
- `predict()` is applied on large chunks without batching.
- No parallelization is used.
- Data frames are not optimized for large-scale operations.

---

**Optimization Strategy:**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.
2. **Batch predictions** to avoid loading entire year subsets into memory at once.
3. **Parallelize predictions** using `parallel` or `future.apply`.
4. **Avoid repeated subsetting of `pred_db`** by merging predictions after processing.
5. **Write results incrementally** or at the end to reduce memory overhead.

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Get unique years
years <- unique(pred_db$year)

# Detect available cores
n_cores <- max(1, detectCores() - 1)

# Parallel prediction
cl <- makeCluster(n_cores)
clusterExport(cl, c("joined_data", "rf_models_per_year"))
clusterEvalQ(cl, library(randomForest))

pred_list <- parLapply(cl, years, function(yr) {
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    test_set <- joined_data[year == yr]
    
    # Batch prediction to avoid memory overload
    batch_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = batch_size)) {
      idx <- i:min(i + batch_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    data.table(id = test_set$id, year = yr, consolidated = preds)
  } else {
    NULL
  }
})

stopCluster(cl)

# Combine predictions
pred_dt <- rbindlist(pred_list, use.names = TRUE, fill = TRUE)

# Merge back into pred_db
pred_db <- merge(pred_db, pred_dt, by = c("id", "year"), all.x = TRUE)

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**
- `data.table` for speed and low memory footprint.
- Parallelization across years using `parLapply`.
- Batch predictions to prevent memory spikes.
- Merge predictions at the end instead of in-loop assignments.
- Incremental writing with `fwrite` for efficiency.

This approach makes the process feasible on a normal machine without retraining models.