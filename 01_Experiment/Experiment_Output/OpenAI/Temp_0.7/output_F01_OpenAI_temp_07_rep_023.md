 **Diagnosis**  
- The bottleneck arises from looping over years and applying `predict()` on large `data.frame` objects with many predictors.  
- `joined_data %>% filter(year == year)` repeatedly copies large data subsets, increasing memory overhead.  
- `predict()` on `randomForest` objects can be slow for large datasets when predictions are done serially.  
- `pred_db$consolidated[...] <- ...` repeated writes on a large data frame are costly.  

---

**Optimization Strategy**  
1. **Use `data.table`** for efficient filtering and assignment instead of `dplyr`.  
2. **Avoid repeated copies**: Split `joined_data` by year once.  
3. **Parallelize predictions** using `parallel::mclapply` or `future.apply`.  
4. **Preallocate output vector** and fill by block to reduce write overhead.  
5. **Use `predict(..., type="response")` directly on `data.table` without conversion to tibble.**  
6. **Do not retrain models; only load and predict.**  

---

**Optimized R Code**  
```r
library(data.table)
library(parallel)

# Load data and models
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Split data by year once
data_by_year <- split(joined_data, by = "year")

# Prepare result vector
pred_db[, consolidated := NA_real_]

# Detect available cores
ncores <- max(1, detectCores() - 1)

# Parallel prediction loop
results <- mclapply(names(data_by_year), function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    test_set <- data_by_year[[yr]]
    preds <- predict(rf_model, newdata = test_set)
    list(year = as.integer(yr), ids = test_set$id, preds = preds)
  } else {
    NULL
  }
}, mc.cores = ncores)

# Combine predictions back
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year & id %in% res$ids, consolidated := res$preds]
  }
}

# Save final output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this is faster/more memory-efficient:**  
- `data.table` avoids expensive copies from `dplyr`.  
- Pre-splitting by year eliminates repeated filtering.  
- Parallel execution utilizes multiple cores for prediction.  
- In-place assignment reduces memory overhead.  

**Expected Impact:**  
- Several-fold speedup, lower memory consumption, and still preserves the trained Random Forest models.