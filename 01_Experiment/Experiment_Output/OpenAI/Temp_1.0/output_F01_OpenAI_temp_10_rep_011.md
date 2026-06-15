 **Diagnosis:**  
- The code loops through each year, filters `joined_data` repeatedly, and predicts row by row, which is inefficient for large datasets.  
- `predict()` on large `data.frame`s can be memory-intensive.  
- `joined_data` is repeatedly subsetted, creating large intermediate objects in memory.  
- `data.frame` operations are slower than optimized formats like `data.table`.  
- No parallelization is used, even though prediction tasks for different years are independent.

---

**Optimization Strategy:**  
1. Convert data to `data.table` for fast filtering and assignment.  
2. Avoid creating large intermediate objects by using in-place operations.  
3. Use `split()` once instead of repetitive filtering.  
4. Add parallelization to process multiple years concurrently if system has multiple cores.  
5. Directly write predictions incrementally or as a combined vector without growing objects inside a loop.

---

**Optimized R Code:**  

```r
library(data.table)
library(parallel)

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

unique_years <- unique(pred_db$year)

# Split joined_data by year once
joined_split <- split(joined_data, by = "year", keep.by = TRUE)

# Parallel prediction
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
clusterExport(cl, c("rf_models_per_year", "joined_split"))
clusterEvalQ(cl, library(randomForest))

predictions <- parLapply(cl, unique_years, function(yr) {
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    test_set <- joined_split[[as.character(yr)]]
    predict(rf_model, newdata = test_set)
  } else {
    rep(NA, nrow(joined_split[[as.character(yr)]]))
  }
})
stopCluster(cl)

# Combine predictions into pred_db
pred_db[, consolidated := unlist(predictions)]

# Save output to CSV
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why This Works Better:**  
- `data.table` drastically speeds up filtering and assignment.  
- `split()` eliminates repeated filtering in the loop.  
- Parallelization distributes workload across cores.  
- `fwrite()` is faster and more memory-efficient than `write.csv()`.  

This approach preserves trained Random Forest models, keeps memory usage manageable, and significantly reduces runtime.