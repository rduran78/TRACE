 **Diagnosis:**  
The bottleneck is the loop-based prediction on large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and base R indexing. This approach repeatedly subsets large data frames and calls `predict()` sequentially, which is memory- and time-intensive.  

**Optimization Strategy:**  
1. **Avoid repeated filtering:** Split `pred_db` and `joined_data` by year once, then process in chunks.  
2. **Use `data.table` for fast subsetting and assignment.**  
3. **Parallelize predictions** using `parallel::mclapply` or `future.apply`.  
4. **Chunk large datasets** to avoid memory overload during `predict()`.  
5. **Write output incrementally** to reduce memory footprint.  

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Ensure year is character for consistent matching
pred_db[, year := as.character(year)]
joined_data[, year := as.character(year)]

# Get list of years present in both datasets
years <- intersect(unique(pred_db$year), names(rf_models_per_year))

# Number of cores for parallel processing
n_cores <- max(1, detectCores() - 1)

# Function to predict for one year in chunks
predict_year <- function(yr) {
  cat("Predicting for year", yr, "\n")
  
  rf_model <- rf_models_per_year[[yr]]
  test_set <- joined_data[year == yr]
  
  # Chunk size (adjust based on memory)
  chunk_size <- 50000
  n <- nrow(test_set)
  preds <- numeric(n)
  
  for (i in seq(1, n, by = chunk_size)) {
    idx <- i:min(i + chunk_size - 1, n)
    preds[idx] <- predict(rf_model, newdata = test_set[idx])
  }
  
  # Return data.table with predictions
  return(data.table(id = test_set$id, pred = preds)) # assumes 'id' exists
}

# Run in parallel
results <- mclapply(years, predict_year, mc.cores = n_cores)

# Combine predictions
pred_dt <- rbindlist(results)

# Merge back into pred_db
pred_db[pred_dt, consolidated := i.pred, on = "id"]

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**  
- **`data.table`** drastically speeds up filtering and assignment.  
- **Parallelization** uses multiple cores for different years.  
- **Chunking** prevents memory overload during `predict()`.  
- **Incremental merging** avoids creating huge intermediate objects.  

This approach preserves the trained models, uses minimal memory, and scales efficiently for large datasets on a personal machine.