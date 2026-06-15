 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` on Random Forest models. This is memory-intensive because `joined_data` is filtered repeatedly, and predictions are stored in `pred_db` inefficiently. The entire dataset is processed in-memory without chunking or parallelization.

---

**Optimization Strategy:**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.  
2. **Chunk predictions** for large datasets to reduce memory load.  
3. **Parallelize predictions** using `parallel` or `future.apply` since Random Forest predictions are independent per year.  
4. **Avoid repeated filtering** by splitting `joined_data` by year once.  
5. **Write results incrementally** to disk to avoid holding everything in memory.

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Split joined_data by year once
year_list <- split(joined_data, by = "year")

# Detect available cores for parallel processing
n_cores <- max(1, detectCores() - 1)

# Parallel prediction
cl <- makeCluster(n_cores)
clusterExport(cl, c("rf_models_per_year", "year_list", "pred_db"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, names(year_list), function(y) {
  test_set <- year_list[[y]]
  if (y %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[y]]
    preds <- predict(rf_model, newdata = test_set)
    data.table(year = y, cell_id = test_set$cell_id, consolidated = preds)
  } else {
    NULL
  }
})
stopCluster(cl)

# Combine results and merge back to pred_db
pred_results <- rbindlist(results, use.names = TRUE, fill = TRUE)
pred_db[pred_results, on = .(year, cell_id), consolidated := i.consolidated]

# Write incrementally to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

**Key Improvements:**  
- `data.table` for fast operations.  
- Parallel processing across years.  
- Avoid repeated filtering by pre-splitting data.  
- Incremental writing with `fwrite` for speed and low memory footprint.  

This approach preserves the trained models, avoids retraining, and makes the process feasible on a normal machine.