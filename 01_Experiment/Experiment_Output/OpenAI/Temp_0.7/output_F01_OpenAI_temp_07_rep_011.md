 **Diagnosis:**  
The main bottlenecks in your script are:  
1. **Row-wise prediction in a loop** — Iterating year by year with large data subsets is inefficient.  
2. **`dplyr` filtering per iteration** — Repeated subsetting is expensive on large data.  
3. **Memory overhead** from handling full `sf` objects; geometry is unnecessary for prediction.  
4. Lack of **parallelization** — Predictions are independent and can be parallelized.  

---

### **Optimization Strategy**
- **Drop geometry**: Convert `sf` object to `data.table` or `data.frame` for prediction.  
- **Use `data.table` joins** for fast filtering instead of `dplyr`.  
- **Vectorize prediction**: Apply `predict()` on each year group efficiently.  
- **Parallelize**: Use `parallel` or `future.apply` to process years concurrently.  
- **Chunking**: If memory is still an issue, process predictions in chunks.  

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Drop geometry if present
if ("geometry" %in% names(joined_data)) joined_data[, geometry := NULL]
if ("geometry" %in% names(pred_db)) pred_db[, geometry := NULL]

# Get unique years
years <- unique(pred_db$year)

# Detect available cores
n_cores <- max(1, detectCores() - 1)

# Parallel prediction by year
cl <- makeCluster(n_cores)
clusterExport(cl, c("rf_models_per_year", "joined_data", "pred_db"), envir = environment())
clusterEvalQ(cl, {
  library(randomForest)
  library(data.table)
})

pred_list <- parLapply(cl, years, function(y) {
  model_name <- as.character(y)
  if (model_name %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[model_name]]
    test_set <- joined_data[year == y]
    preds <- predict(rf_model, newdata = test_set)
    data.table(year = y, id = test_set$id, pred = preds)
  } else {
    data.table(year = y, id = integer(0), pred = numeric(0))
  }
})

stopCluster(cl)

# Combine predictions and join back
pred_all <- rbindlist(pred_list)
setkey(pred_all, year, id)
setkey(pred_db, year, id)
pred_db <- pred_db[pred_all, consolidated := i.pred]

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Key Gains**
- **Parallel execution** reduces runtime drastically on multi-core machines.  
- **Avoid repeated filtering** by using `data.table` joins.  
- **Memory efficiency** by removing geometry and using `fwrite` instead of `write.csv`.  

This approach preserves the trained models and makes the process feasible on a personal computer.