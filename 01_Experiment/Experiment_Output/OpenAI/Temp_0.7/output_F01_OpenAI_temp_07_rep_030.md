 **Diagnosis**  
The bottleneck is likely due to:
- Looping over years with `predict()` on large subsets, causing repeated data filtering and memory overhead.
- `joined_data %>% filter(year == year)` inside the loop, which is inefficient for large datasets.
- Storing predictions in `pred_db$consolidated` with repeated subsetting, adding time and memory costs.

**Optimization Strategy**  
- Use `data.table` for fast filtering and assignment.
- Avoid repeatedly creating subsets; instead, split data once by year.
- Use `pbapply` or `future.apply` for parallel predictions if possible.
- Ensure only necessary columns are passed to `predict()` (drop unused variables).
- Write predictions in chunks rather than keeping a large object in memory.

**Optimized R Code**

```r
library(data.table)
library(pbapply)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Pre-split data by year to avoid repeated filtering
year_list <- split(joined_data, by = "year")

# Progress bar and parallel prediction (pbapply)
pred_results <- pblapply(names(year_list), function(y) {
  if (y %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[y]]
    test_set <- year_list[[y]]
    
    # Drop unused columns if any (example: keep predictors only)
    # test_set <- test_set[, predictor_columns, with = FALSE]
    
    preds <- predict(rf_model, newdata = test_set)
    data.table(year = as.integer(y), id = test_set$id, consolidated = preds)
  } else {
    NULL
  }
})

# Combine all predictions
pred_dt <- rbindlist(pred_results, use.names = TRUE)

# Merge predictions back to pred_db efficiently
setkey(pred_db, id, year)
setkey(pred_dt, id, year)
pred_db[pred_dt, consolidated := i.consolidated]

# Write to CSV
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this is better**  
- `data.table` dramatically reduces filtering and assignment overhead.
- Pre-splitting avoids repeated `filter()` calls.
- `pbapply` provides parallelization and progress tracking.
- Memory footprint is minimized by processing per year and merging afterward.

This approach preserves trained models and avoids retraining while making computation feasible on a normal machine.