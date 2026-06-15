 **Diagnosis**  
The bottleneck is the large data size and the year-by-year loop performing predictions serially in memory. RandomForest `predict()` can handle batch predictions relatively efficiently, but looping in R combined with `dplyr` filtering per year incurs heavy overhead. Additionally, loading the full dataset into memory and copying subsets for each year wastes memory and time.

---

**Optimization Strategy**  
1. **Avoid repeated filtering:** Split or index the data by year beforehand.  
2. **Use `data.table` for subsetting & assignment:** Faster and memory-efficient.  
3. **Vectorize predictions if possible:** Predict for all rows grouped by model-year rather than per row.  
4. **Enable parallelization:** Use `parallel` and `foreach` to predict in parallel over years.  
5. **Write results incrementally:** Avoid holding everything in memory until the end.

---

**Optimized R Code**  
```r
library(data.table)
library(foreach)
library(doParallel)

# Convert input data to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Detect available cores and register parallel backend
n_cores <- max(1, parallel::detectCores() - 1)
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Create an index by year for quick joins
joined_data_split <- split(joined_data, by = "year")

# Prepare result column
pred_db[, consolidated := NA_real_]

# Parallel prediction per year
foreach(y = names(rf_models_per_year), .packages = c("data.table", "randomForest")) %dopar% {
  rf_model <- rf_models_per_year[[y]]
  test_set <- joined_data_split[[y]]
  if (!is.null(test_set)) {
    preds <- predict(rf_model, newdata = test_set)
    pred_db[year == as.integer(y), consolidated := preds]
  }
}

stopCluster(cl)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` avoids costly `dplyr` pipeline overhead.  
- Splitting data once and using fast joins reduces repeated subsetting cost.  
- Parallelizing with `foreach` reduces runtime significantly on multi-core machines.  
- Avoids retraining models, strictly uses `predict()`.  

This approach is practical for a personal computer while preserving original trained models and ensuring scalability.