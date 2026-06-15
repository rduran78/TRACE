 **Diagnosis:**  
The current implementation uses a loop with `predict()` for each year, handling large `data.frame`s repeatedly in memory. This is inefficient because:  
- `dplyr::filter()` in a loop repeatedly subsets large data.  
- Predictions are written into `pred_db` element by element, creating overhead.  
- No batching or parallelization is used.  
- Uses base R structures instead of memory-efficient `data.table` or chunking.  

---

**Optimization Strategy:**  
1. Convert large data to `data.table` for fast filtering and assignment.  
2. Avoid repeated filtering by splitting the dataset by year upfront.  
3. Perform predictions in parallel across years using `future.apply` or `parallel`.  
4. Preserve Random Forest models as requested (no retraining).  
5. Combine predictions efficiently at the end.  
6. Optionally chunk within each year if memory is still tight.  

---

**Optimized R Code:**
```r
library(data.table)
library(future.apply)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_by_year <- split(joined_data, joined_data$year)
years <- names(data_by_year)

# Parallel setup (adjust workers as per cores available)
plan(multisession, workers = parallel::detectCores() - 1)

# Predict in parallel per year
results_list <- future_lapply(years, function(yr) {
  test_set <- data_by_year[[yr]]
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    preds <- predict(rf_model, newdata = test_set)
    data.table(year = as.integer(yr),
               id = test_set$id,  # assuming there's an 'id' column
               consolidated = preds)
  } else {
    data.table(year = as.integer(yr), id = test_set$id, consolidated = NA)
  }
})

# Combine and merge predictions back to pred_db
predictions_dt <- rbindlist(results_list)
setkey(predictions_dt, id, year)
setkey(pred_db, id, year)
pred_db <- predictions_dt[pred_db]

# Write in one go
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` dramatically reduces filtering/assignment overhead.  
- One-time split avoids repeated subsetting.  
- Parallelization uses CPU cores effectively for per-year predictions.  
- Batch writing (`fwrite`) avoids memory bottlenecks.  

If RAM issues persist, add **in-year chunking**:
```r
chunk_size <- 50000
chunks <- split(test_set, (seq_len(nrow(test_set)) - 1) %/% chunk_size)
```
and predict per chunk.  
This makes the process computationally feasible on a normal machine without retraining models.