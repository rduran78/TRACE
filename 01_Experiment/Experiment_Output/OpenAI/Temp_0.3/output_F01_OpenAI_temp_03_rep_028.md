 **Diagnosis**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` for Random Forest models. Each iteration loads a large subset (`test_set`) and processes it sequentially, which is memory-intensive and slow. Additionally, `joined_data` and `pred_db` are likely large `data.frame`s, and repeated filtering adds overhead.

**Optimization Strategy**  
- Use `data.table` for efficient filtering and assignment instead of `dplyr`.
- Avoid creating large intermediate objects repeatedly; process in chunks.
- Leverage parallelization for predictions across years.
- Ensure only necessary columns are passed to `predict()` to reduce memory footprint.
- Write results incrementally to disk instead of holding everything in memory.

**Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Identify predictor columns (exclude non-feature columns like 'year')
predictor_cols <- setdiff(names(joined_data), "year")

# Function for prediction per year
predict_year <- function(year) {
  message("Predicting for year ", year)
  test_set <- joined_data[year == !!year, ..predictor_cols]
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    preds <- predict(rf_model, newdata = test_set)
    return(data.table(year = year, preds = preds))
  } else {
    warning("No model found for year ", year)
    return(NULL)
  }
}

# Parallel prediction
years <- unique(pred_db$year)
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("joined_data", "rf_models_per_year", "predictor_cols", "predict_year"))
clusterEvalQ(cl, library(randomForest))
results <- parLapply(cl, years, predict_year)
stopCluster(cl)

# Combine predictions
pred_dt <- rbindlist(results, use.names = TRUE, fill = TRUE)

# Merge back into pred_db efficiently
setkey(pred_db, year)
setkey(pred_dt, year)
pred_db[pred_dt, consolidated := i.preds]

# Write output to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

**Key Improvements**  
- `data.table` for fast filtering and assignment.
- Parallelization using `parLapply` to utilize multiple cores.
- Reduced memory usage by selecting only predictor columns.
- Efficient merging and disk writing with `fwrite`.

This approach preserves trained models, avoids retraining, and makes the process feasible on a standard machine.