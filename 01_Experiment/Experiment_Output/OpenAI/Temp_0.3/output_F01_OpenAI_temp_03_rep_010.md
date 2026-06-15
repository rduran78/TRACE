 **Diagnosis:**  
The main bottleneck is the loop predicting year by year on large datasets in memory. Each iteration filters `joined_data` and writes predictions into `pred_db`. This is inefficient because:  
- `dplyr::filter()` repeatedly scans the full dataset.  
- Predictions are computed sequentially without batching or parallelization.  
- Large objects (`joined_data`, `pred_db`) remain in memory.  

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment.  
- Avoid repeated subsetting by splitting data once.  
- Use `predict()` in parallel with `future.apply` or `parallel`.  
- Write results incrementally to disk to reduce memory pressure.  
- Keep models in memory but process data in chunks.  

---

### **Optimized R Code**

```r
library(data.table)
library(future.apply)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, joined_data$year)

# Prepare parallel backend
plan(multisession, workers = max(1, parallel::detectCores() - 1))

# Function to predict for one year
predict_for_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- data_by_year[[as.character(year)]]
    preds <- predict(rf_model, newdata = test_set)
    return(data.table(year = year, id = test_set$id, pred = preds))
  } else {
    warning(paste("No model found for year", year))
    return(NULL)
  }
}

# Run predictions in parallel
years <- names(data_by_year)
results_list <- future_lapply(years, predict_for_year)

# Combine all predictions
all_preds <- rbindlist(results_list, use.names = TRUE, fill = TRUE)

# Merge predictions back into pred_db by id and year
setkey(pred_db, id, year)
setkey(all_preds, id, year)
pred_db[all_preds, consolidated := i.pred]

# Write to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically speeds up filtering and merging.  
- `future_lapply` parallelizes predictions across years.  
- Splitting once avoids repeated filtering.  
- Writing with `fwrite` is faster and memory-efficient.  

This approach preserves trained models, avoids retraining, and makes the process feasible on a normal machine.