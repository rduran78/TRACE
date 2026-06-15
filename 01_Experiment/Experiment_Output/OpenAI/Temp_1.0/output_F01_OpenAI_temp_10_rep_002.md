 **Diagnosis:**
- The bottleneck comes from looping over years and repeatedly filtering `joined_data`, then predicting on large subsets.
- `predict()` on Random Forest models for hundreds of thousands of rows can be memory-intensive if done in bulk.
- Entire data is held in memory (`pred_db` plus `joined_data`), stressing RAM.

---

**Optimization Strategy:**
1. **Avoid repeated filtering:** Use data.table for fast subsetting and assignment.
2. **Chunk predictions:** Process large datasets in manageable blocks to reduce memory usage.
3. **Vectorized assignments:** Instead of row-wise updates, work by index or merge results back.
4. **Minimize copies:** Disable unnecessary `dplyr` pipes within loops.

---

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

years <- unique(pred_db$year)

# Function to predict in chunks
chunk_predict <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  starts <- seq(1, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1, n)
    preds[s:e] <- predict(model, newdata = data[s:e])
  }
  preds
}

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    idx <- pred_db$year == yr
    test_set <- joined_data[idx]
    pred_db[idx, consolidated := chunk_predict(rf_model, test_set, chunk_size = 50000)]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why faster and lighter?**
- `data.table` reduces overhead in filtering and assigning.
- Chunking prediction prevents `predict()` from loading all rows into memory.
- Minimal data copying compared to `dplyr`.

**Additional tips:**
- Increase `chunk_size` if RAM allows, decrease for very constrained systems.
- Consider compressing intermediate objects (`rm()` + `gc()` post-loop).
- If still slow, parallelize chunks using `parallel::mclapply()` (Linux/Mac) or `foreach` with a parallel backend.