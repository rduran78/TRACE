 **Diagnosis:**  
The current bottleneck comes from:
- Loading and processing large datasets entirely in memory.
- Looping over years with full `joined_data` filtering for each iteration (costly with hundreds of thousands of rows).
- Using `predict()` on huge chunks without batching or parallelization.
- Writing the full `pred_db` only at the end, increasing memory usage.

---

**Optimization Strategy:**  
1. **Use `data.table` for efficient filtering and assignment** instead of `dplyr`.
2. **Process data in chunks** (batch predictions) to reduce memory load.
3. **Parallelize predictions** using `parallel::mclapply` or `future.apply`.
4. **Avoid copying large data frames repeatedly**—filter and predict in-place.
5. **Incremental write to disk** instead of holding everything in memory until the end.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Detect available cores for parallel
n_cores <- max(1, detectCores() - 1)

# Function for chunked prediction
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  return(preds)
}

# Open a file connection for incremental writing
fwrite(pred_db[0], "RF_imputated_db.csv")  # write header first

# Loop over years in parallel
mclapply(unique(pred_db$year), function(year) {
  message("Predicting for year ", year)
  
  subset_rows <- pred_db$year == year
  test_set <- joined_data[year == !!year]
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Chunked prediction
    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000)
    
    # Assign predictions
    pred_db[subset_rows, consolidated := preds]
    
    # Incremental write
    fwrite(pred_db[subset_rows], "RF_imputated_db.csv", append = TRUE)
  } else {
    warning("No model found for year ", year)
  }
}, mc.cores = n_cores)

message("Prediction complete.")
```

---

**Why this is practical:**
- `data.table` drastically reduces memory overhead vs. `dplyr`.
- Chunked prediction avoids loading huge matrices at once.
- Parallelization speeds up year-wise processing.
- Incremental writing prevents memory overflow.

This approach preserves your trained models and makes prediction feasible on a standard machine.