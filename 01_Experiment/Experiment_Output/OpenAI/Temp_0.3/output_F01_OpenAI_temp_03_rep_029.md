 **Diagnosis**  
The bottleneck is the `for` loop predicting year by year on large datasets using `predict()` for Random Forest models. This approach is slow and memory-intensive because:  
- It repeatedly subsets large `data.frame`s (`joined_data`) for each year.  
- Predictions are done sequentially without leveraging vectorization or efficient data structures.  
- `pred_db` is updated in a non-vectorized way.  

**Optimization Strategy**  
1. **Convert to `data.table`** for fast filtering and assignment.  
2. **Avoid repeated subsetting** by splitting the data once by year.  
3. **Parallelize predictions** using `parallel::mclapply` or `future.apply`.  
4. **Preallocate and merge results** instead of updating in place.  
5. **Use `predict(..., type="response")` efficiently** without copying large objects repeatedly.  

**Optimized R Code**  
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_split <- split(joined_data, by = "year")

# Get available models
model_years <- names(rf_models_per_year)

# Function to predict for one year
predict_year <- function(yr) {
  if (yr %in% model_years) {
    rf_model <- rf_models_per_year[[yr]]
    dt <- data_split[[yr]]
    preds <- predict(rf_model, newdata = dt)
    return(data.table(year = dt$year, cell_id = dt$cell_id, consolidated = preds))
  } else {
    warning(paste("No model found for year", yr))
    return(NULL)
  }
}

# Parallel prediction
years <- names(data_split)
ncores <- max(1, detectCores() - 1)
pred_list <- mclapply(years, predict_year, mc.cores = ncores)

# Combine predictions
pred_results <- rbindlist(pred_list, use.names = TRUE, fill = TRUE)

# Merge back into pred_db
pred_db <- merge(pred_db, pred_results, by = c("year", "cell_id"), all.x = TRUE)
setnames(pred_db, "consolidated.y", "consolidated")
pred_db[, consolidated.x := NULL]

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works**  
- `data.table` drastically reduces memory overhead and speeds up filtering/merging.  
- Splitting once avoids repeated heavy operations.  
- Parallelization uses multiple cores for prediction.  
- Merging results is more efficient than in-loop assignment.  

This approach will make the process much faster and memory-friendly without retraining models.