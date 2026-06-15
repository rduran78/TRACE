 **Diagnosis**  
- The bottleneck comes from **predicting in a loop over years** on large data frames without efficient memory handling.  
- `predict()` on `randomForest` models with huge `newdata` consumes significant RAM, especially when `test_set` is large.  
- `joined_data %>% filter()` creates a full copy each iteration, adding overhead.  
- The process is single-threaded and does not leverage chunking or memory-efficient structures.  

---

**Optimization Strategy**  
1. **Avoid repeated filtering**: Use `split()` once to partition `joined_data` by `year`.  
2. **Predict in chunks**: Break large test sets into smaller batches to reduce memory spikes.  
3. **Use `data.table`**: Faster filtering and assignment than `dplyr`.  
4. **Parallelize predictions**: Use `future.apply` or `parallel` for multi-core execution if possible.  
5. **Write incrementally**: Reduce memory pressure by writing results to disk progressively.  

---

**Working Optimized R Code**  

```r
library(data.table)
library(future.apply)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Split data by year once
year_splits <- split(joined_data, by = "year", keep.by = FALSE)

# Prepare parallel plan (adjust workers based on CPU cores)
plan(multisession, workers = max(1, parallel::detectCores() - 1))

# Function to predict in chunks
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  preds
}

# Loop over years in parallel
results <- future_lapply(names(year_splits), function(yr) {
  message("Predicting for year ", yr)
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    data_yr <- year_splits[[yr]]
    preds <- predict_in_chunks(rf_model, data_yr)
    data.table(id = data_yr$id, year = as.integer(yr), consolidated = preds)
  } else {
    warning("No model found for year ", yr)
    NULL
  }
})

# Combine results
results <- rbindlist(results, use.names = TRUE, fill = TRUE)

# Merge predictions back into pred_db
pred_db[results, on = .(id, year), consolidated := i.consolidated]

# Write output to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Benefits**  
- **Chunked predictions** prevent memory overload.  
- **Parallelization** speeds up multi-year predictions.  
- **Data.table** provides high performance for filtering and merging.  
- Preserves existing trained `rf_models_per_year`.  

This approach makes the process feasible on a normal machine without retraining models.