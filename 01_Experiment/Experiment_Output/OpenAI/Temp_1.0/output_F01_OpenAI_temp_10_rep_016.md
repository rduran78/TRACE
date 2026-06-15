 **Diagnosis**  
The current loop predicts for each year using large data frames and repeatedly calls `predict` with entire subsets in memory. This approach is inefficient because:  
- `test_set` can be huge for some years, stressing RAM.  
- The `predict` call on the entire subset without chunking causes memory bloat and slows computation.  
- Appending predictions inside a loop to `pred_db` via indexing is costly for large datasets.  

**Optimization Strategy**  
- Use `data.table` for faster filtering and assignment.  
- Process predictions in chunks (batching) to prevent memory overload.  
- Avoid repeatedly subsetting `pred_db` inside the loop; instead, compute predictions and assign at the end using row ordering.  
- Use only required columns during prediction.  
- Parallelize across available cores if possible.  

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Ensure a key column for merging results quickly
pred_db[, pred_id := .I]
joined_data[, pred_id := .I]

# Identify years and available models
years <- unique(pred_db$year)
n_cores <- detectCores() - 1  # leave 1 core free

# Function for chunked prediction
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  starts <- seq(1, n, by = chunk_size)
  for (i in starts) {
    end <- min(i + chunk_size - 1, n)
    chunk <- data[i:end]
    preds[i:end] <- predict(model, newdata = chunk)
  }
  return(preds)
}

# Parallel execution by year
cl <- makeCluster(n_cores)
clusterExport(cl, c("rf_models_per_year", "joined_data", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

results_list <- parLapply(cl, years, function(y) {
  if (as.character(y) %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[as.character(y)]]
    subset_data <- joined_data[year == y]
    preds <- predict_in_chunks(model, subset_data)
    data.table(pred_id = subset_data$pred_id, consolidated = preds)
  } else {
    NULL
  }
})
stopCluster(cl)

# Combine results and merge back
results_dt <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
pred_db <- merge(pred_db, results_dt, by = "pred_id", all.x = TRUE)
pred_db[, pred_id := NULL]

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why It Works**  
- `data.table` speeds up filtering and assignment.  
- Processing predictions in **50k-row chunks** prevents memory overflow.  
- Parallelization utilizes multiple cores to handle different years.  
- Preserves trained Random Forest models exactly as required.  

This approach makes the computation feasible on a standard personal machine without retraining models.