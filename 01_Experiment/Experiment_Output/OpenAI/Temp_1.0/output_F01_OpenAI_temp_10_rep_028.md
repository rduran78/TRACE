 **Diagnosis**  
- The loop applies prediction year-by-year using `dplyr::filter`, creating large intermediate data frames repeatedly → inefficient.  
- `predict()` runs on possibly hundreds of thousands of rows with many variables at once, increasing memory load.  
- Use of `joined_data` (likely large) in repetitive subsetting and copy operations exacerbates slowness.  
- Writing `pred_db` row-by-row indirectly (by assignment in loop) creates expensive vector updates.  

---

**Optimization Strategy**  
1. **Avoid expensive filtering in loop**: Split `joined_data` by year once instead of filtering repeatedly.  
2. **Vectorize assignment**: Combine predictions into a separate vector, then join back in one step.  
3. **Use `data.table`** for efficient filtering and in-place updates.  
4. **Batch predict**: Predict per year, but operate on pre-split sets and return results as list, then combine.  
5. **Reduce memory copies**: Do not repeatedly copy `pred_db`; maintain predictions separately and merge after.  
6. **Parallelization**: Utilize `parallel` or `future.apply` when predicting by year to use multiple cores.  

---

**Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Pre-split data by year
year_splits <- split(joined_data, joined_data$year)

# Detect available cores for parallel processing
n_cores <- max(1, detectCores() - 1)

# Parallel prediction by year
pred_list <- mclapply(names(year_splits), function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    preds <- predict(rf_model, newdata = year_splits[[yr]])
    data.table(id = year_splits[[yr]]$id, year = as.integer(yr), pred = preds)
  } else {
    NULL
  }
}, mc.cores = n_cores)

# Combine all predictions into single data.table
pred_dt <- rbindlist(pred_list, use.names = TRUE, fill = TRUE)

# Merge predictions back into pred_db
pred_db <- merge(pred_db, pred_dt, by = c("id", "year"), all.x = TRUE)
pred_db$consolidated <- pred_db$pred
pred_db$pred <- NULL

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why This Works**  
- `data.table` and pre-splitting drastically cut down on repeated filtering.  
- `mclapply` runs predictions in parallel, using all available cores.  
- Output writing is efficient via `fwrite`.  
- No retraining occurs; only prediction step is streamlined.  

This approach makes it computationally feasible for very large datasets on a standard machine.