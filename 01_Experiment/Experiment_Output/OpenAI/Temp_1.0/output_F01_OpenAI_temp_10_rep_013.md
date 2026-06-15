 **Diagnosis:**  
The current loop processes the entire dataset year by year, each time subsetting large data (`joined_data`) and predicting in-memory. With hundreds of thousands of rows and many predictors, memory and speed become bottlenecks. `predict()` on large data with Random Forest can be slow, and the full `test_set` for each year increases strain. Additionally, the process is entirely single-threaded and lacks chunking.  

---

**Optimization Strategy:**  
- **Use `data.table` for fast filtering and assignment** instead of `dplyr`.  
- **Perform prediction in chunks** (batch predict large datasets) to reduce memory spikes.  
- **Parallelize prediction** using multiple cores with `parallel` or `future.apply` if possible.  
- **Avoid unnecessary copies** by preallocating result vectors and working on subsets without additional joins.  

---

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Ensure matching key columns for fast filtering
setkey(pred_db, year)
setkey(joined_data, year)

# Identify available cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Define chunked prediction function
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  for (i in seq(1, n, by = chunk_size)) {
    idx <- i:min(i + chunk_size - 1, n)
    preds[idx] <- predict(model, newdata[idx, ])
  }
  return(preds)
}

# Process each year
years <- unique(pred_db$year)

# Parallelize across years
mclapply(years, function(year) {
  cat(paste0("Predicting for year ", year, "\n"))
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[J(year)]
    
    if (nrow(test_set) > 0) {
      preds <- predict_in_chunks(rf_model, test_set)
      pred_db[year == !!year, consolidated := preds]
    }
  } else {
    cat(paste0("Warning: No model found for year ", year, "\n"))
  }
  
  NULL
}, mc.cores = n_cores)

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
1. Uses `data.table` for efficient subsetting and assignment.  
2. Predicts in manageable chunks (`chunk_size` adjustable) to prevent memory overload.  
3. Parallelizes by year to utilize multiple cores on normal hardware.  
4. Avoids full data copying inside the loop.  

This approach makes the process significantly faster and memory-friendly without retraining models.