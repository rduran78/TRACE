 **Diagnosis:**  
The bottleneck stems from:
- Looping over years and predicting on large `test_set` subsets sequentially.
- Using `dplyr` filtering repeatedly on a large dataset (`joined_data`), which is memory- and time-intensive.
- No batching or parallelization for large predictions.
- Storing and writing big objects without efficient formats.

---

**Optimization Strategy:**  
1. **Pre-split `joined_data` by year** to avoid repeated filtering.
2. Use **`data.table`** for fast filtering and assignment instead of `dplyr`.
3. Perform **batched predictions** for large subsets to reduce memory pressure.
4. Optionally enable **parallel prediction** using `parallel::mclapply` or similar (if RF model supports it).
5. Write output using **`fwrite()`** (fast I/O) instead of `write.csv`.

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Pre-split indices by year
year_groups <- split(seq_len(nrow(joined_data)), joined_data$year)

# Detect available cores
n_cores <- max(1, detectCores() - 1)

# Parallel prediction
results <- mclapply(names(year_groups), function(year) {
  idx <- year_groups[[year]]
  
  if (year %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[year]]
    
    # Batch prediction to avoid memory overload
    batch_size <- 50000
    preds <- numeric(length(idx))
    
    for (i in seq(1, length(idx), by = batch_size)) {
      batch_idx <- idx[i:min(i + batch_size - 1, length(idx))]
      preds[seq_along(batch_idx) + i - 1] <- predict(rf_model, newdata = joined_data[batch_idx])
    }
    
    list(year = year, idx = idx, preds = preds)
  } else {
    warning(paste("No model found for year", year))
    NULL
  }
}, mc.cores = n_cores)

# Assign predictions back to pred_db
for (res in results) {
  if (!is.null(res)) {
    pred_db$consolidated[res$idx] <- res$preds
  }
}

# Fast write
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**
- `data.table` drastically reduces filtering/assignment overhead.
- Pre-splitting avoids repeated `filter()`.
- Prediction done in **parallel** and in **chunks** keeps memory usage under control.
- `fwrite()` speeds up storage for big data.

This approach will make predictions feasible on a personal machine while preserving trained Random Forest models.