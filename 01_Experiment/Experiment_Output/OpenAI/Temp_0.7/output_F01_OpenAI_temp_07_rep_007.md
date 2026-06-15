 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large subsets (`test_set`). Each iteration filters `joined_data` and applies `predict` on large data frames, which is memory- and time-intensive. Since models are pre-trained, the process can be optimized by:  

- **Avoid repeated filtering:** Avoid creating large intermediate `test_set` objects repeatedly.  
- **Chunk predictions:** Split large datasets into smaller chunks to reduce memory load.  
- **Use `data.table` for speed:** Efficient filtering and assignment.  
- **Parallelization:** Use multiple cores to run predictions in parallel.  

---

### **Optimization Strategy**
1. Convert `pred_db` and `joined_data` to `data.table` for efficient operations.  
2. Process chunks per year in parallel using `parallel::mclapply` (Linux/Mac) or `future.apply` (cross-platform).  
3. Predict in smaller batches (e.g., 50k rows) to avoid memory overload.  
4. Write incremental results to disk or use `fwrite` for speed.  

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# Function to predict in chunks
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(1:n, ceiling(seq_along(1:n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  return(preds)
}

# Parallel prediction by year
years <- unique(pred_db$year)
num_cores <- max(1, detectCores() - 1)

results <- mclapply(years, function(year) {
  cat("Predicting for year", year, "\n")
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict_in_chunks(rf_model, test_set)
    return(data.table(year = year, idx = which(pred_db$year == year), pred = preds))
  } else {
    cat("Warning: No model for year", year, "\n")
    return(NULL)
  }
}, mc.cores = num_cores)

# Combine results
results <- rbindlist(results)
pred_db[results$idx, consolidated := results$pred]

# Save efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

✅ **Why this works:**  
- Uses `data.table` for fast filtering and assignment.  
- Splits large predictions into manageable chunks.  
- Leverages parallel processing across years.  
- Avoids creating large intermediate objects repeatedly.  

This approach makes the process **computationally feasible on a normal machine** without retraining models.