 **Diagnosis:**  
The main bottleneck is the loop predicting year by year on large `joined_data` and writing results into `pred_db`. The issues:  
- `dplyr::filter` and `predict` on large subsets repeatedly is slow.  
- `pred_db$consolidated[...] <- ...` inside a loop is memory-expensive.  
- No parallelization is used.  

**Optimization Strategy:**  
1. Convert data to `data.table` for efficient filtering and assignment.  
2. Pre-split data by year in memory to avoid repeated filtering overhead.  
3. Use `parallel::mclapply` or `future.apply` to parallelize predictions.  
4. Avoid growing objects inside the loop—preallocate.  
5. Write output once at the end.  

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# Pre-split joined_data by year
split_joined <- split(joined_data, by = "year")
split_pred <- split(pred_db, by = "year")

years <- names(split_joined)

# Function for prediction
predict_year <- function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    preds <- predict(rf_model, newdata = split_joined[[yr]])
    split_pred[[yr]][, consolidated := preds]
  } else {
    message("Warning: No model found for year ", yr)
  }
  return(split_pred[[yr]])
}

# Parallel prediction
cl <- detectCores() - 1
results <- mclapply(years, predict_year, mc.cores = cl)

# Combine back
pred_db_optimized <- rbindlist(results)

# Write output once
fwrite(pred_db_optimized, "RF_imputated_db.csv")
```

**Why it’s better:**  
- `data.table` dramatically reduces memory and filtering cost.  
- Data is split once, not repeatedly.  
- Parallelization uses all CPU cores.  
- Avoids incremental writes and reallocations.  

This approach preserves trained models, speeds up processing, and is feasible on a personal computer.