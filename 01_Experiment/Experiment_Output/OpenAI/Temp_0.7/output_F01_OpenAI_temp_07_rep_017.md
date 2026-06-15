 **Diagnosis**  
- The bottleneck is the prediction loop over years with large datasets.  
- `predict()` on `randomForest` objects is vectorized, but here it’s applied in a loop without chunking or parallelization.  
- `joined_data` likely contains many columns; subsetting and copying repeatedly increases memory usage.  
- Writing the entire `pred_db` after the loop may cause memory strain.  

---

**Optimization Strategy**  
1. **Chunk predictions**: Break large `test_set` into smaller chunks to keep memory usage low.  
2. **Parallel processing**: Use multiple cores for independent yearly predictions.  
3. **Minimal columns**: Pass only required predictors to `predict()`.  
4. **In-place updates**: Avoid creating large intermediate objects.  
5. **Efficient I/O**: Use `data.table::fwrite()` instead of `write.csv()` for speed.  

---

**Optimized R Code**  

```r
library(data.table)
library(parallel)

# Convert pred_db and joined_data to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Detect cores for parallel processing
n_cores <- max(1, detectCores() - 1)

# Function to predict for one year in chunks
predict_year <- function(year, rf_model, data, chunk_size = 50000) {
  test_set <- data[year == !!year]
  
  if (nrow(test_set) == 0) return(NULL)
  
  # Only keep predictor columns used in the model
  predictors <- names(rf_model$forest$xlevels)
  test_set <- test_set[, ..predictors]
  
  n <- nrow(test_set)
  preds <- numeric(n)
  
  for (i in seq(1, n, by = chunk_size)) {
    idx <- i:min(i + chunk_size - 1, n)
    preds[idx] <- predict(rf_model, newdata = test_set[idx])
  }
  
  data.table(year = year, row_id = which(joined_data$year == year), pred = preds)
}

# Parallel prediction across years
years <- unique(pred_db$year)
results <- mclapply(years, function(y) {
  if (as.character(y) %in% names(rf_models_per_year)) {
    predict_year(y, rf_models_per_year[[as.character(y)]], joined_data)
  } else {
    cat("Warning: No model for year", y, "\n")
    NULL
  }
}, mc.cores = n_cores)

# Combine and update pred_db
results <- rbindlist(results, use.names = TRUE, fill = TRUE)
pred_db[results$row_id, consolidated := results$pred]

# Efficient write
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- **Chunking** avoids blowing up memory during `predict()`.  
- **Parallelization** speeds up processing across years.  
- Using `data.table` and `fwrite()` drastically reduces overhead.  
- Preserves original trained models as requested.