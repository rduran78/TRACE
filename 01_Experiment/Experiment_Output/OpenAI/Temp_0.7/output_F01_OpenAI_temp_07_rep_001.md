 **Diagnosis:**  
The bottleneck is the loop applying `predict()` year by year on large data frames (`joined_data` and `pred_db`), combined with R's in-memory operations. Each iteration extracts subsets and writes predictions back, which is slow and memory intensive for hundreds of thousands of rows.  

**Optimization Strategy:**  
- Avoid repeated `filter()` by splitting data once and processing in chunks.  
- Use `data.table` for efficient subsetting and assignment.  
- Parallelize `predict()` calls across years using `parallel::mclapply` (or `future.apply` on Windows).  
- Preallocate and join results after prediction to minimize in-loop writes.  
- Only keep necessary columns for prediction (`model.matrix` or `predictors only`) to reduce memory footprint.  

---

### **Optimized Working Code**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Keep only columns required by Random Forest models
# Assuming all models use same features; adjust as needed
predictor_cols <- setdiff(names(joined_data), c("year", "target_var"))  # remove non-predictors

# Split joined_data by year
data_split <- split(joined_data[, ..predictor_cols], joined_data$year)
years <- names(data_split)

# Parallel prediction
n_cores <- max(1, detectCores() - 1)

pred_list <- mclapply(years, function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    preds <- predict(rf_model, newdata = data_split[[yr]])
    data.table(year = as.integer(yr), prediction = preds)
  } else {
    warning(paste("No model found for year", yr))
    NULL
  }
}, mc.cores = n_cores)

# Combine predictions
pred_dt <- rbindlist(pred_list, use.names = TRUE, fill = TRUE)

# Merge back into pred_db
setkey(pred_db, year)
setkey(pred_dt, year)
pred_db[pred_dt, consolidated := prediction]

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically speeds up filtering and assignment.  
- Splitting once and reusing chunks avoids repeated filtering.  
- Parallelization leverages multiple cores for independent predictions.  
- Memory usage is reduced by working with essential columns only and combining results at the end.  

This approach should make the process feasible on a personal machine without retraining models.