 **Diagnosis:**  
- The bottleneck comes from looping through each year and applying `predict()` on large `data.frame`s in memory.  
- `predict.randomForest()` can be slow for very large datasets because it processes the entire dataset in one shot.  
- Memory footprint grows when repeatedly creating large intermediate objects (`test_set`, filtered `data.frame`s).  

**Optimization Strategy:**  
- Avoid large in-memory operations:
  - **Use `data.table`** for efficient subsetting and assignment instead of `dplyr`.
- **Batch predictions (chunked processing)** to avoid loading entire year blocks at once.
- **Parallelize** predictions across cores using `parallel::mclapply` or similar if available.
- Keep Random Forest models as they are and process in blocks without retraining.

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for fast operations
setDT(pred_db)
setDT(joined_data)

# Identify available cores
n_cores <- max(1, detectCores() - 1)

# Chunk size for batch prediction (adjust based on machine memory)
chunk_size <- 5000  

# Function to predict in batches
predict_in_chunks <- function(model, newdata, chunk_size) {
  n <- nrow(newdata)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = newdata[idx[[i]], ])
  }
  preds
}

# Main optimized loop using parallel processing per year
years <- unique(pred_db$year)

mclapply(years, function(year) {
  cat("Predicting for year", year, "\n")
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == ..year]  # fast filter
    preds <- predict_in_chunks(rf_model, test_set, chunk_size)
    pred_db[year == ..year, consolidated := preds]
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}, mc.cores = n_cores)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**
- **`data.table`** replaces `dplyr` for fast filtering and assignment.
- **Chunked prediction** avoids memory overload.
- **Parallel computation per year** reduces runtime significantly on multi-core CPUs.
- No retraining, models are used as-is.

This approach should make it practical to run on a normal machine without exhausting memory.