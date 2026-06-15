 **Diagnosis:**  
The bottleneck is the loop over years combined with `predict()` on large `test_set` objects in memory. Each iteration filters `joined_data` and assigns predictions to `pred_db`, which is inefficient for hundreds of thousands of rows. Additionally, `dplyr` operations and repeated subsetting add overhead. The process is single-threaded and does not exploit chunking or memory-efficient data structures.

---

**Optimization Strategy:**  
1. **Use `data.table` for filtering and assignment** (faster than `dplyr` for large data).
2. **Predict in chunks** to avoid loading the entire year's data into memory at once.
3. **Parallelize predictions** using `parallel` or `future.apply` if multiple cores are available.
4. **Avoid repeated subsetting of `pred_db`** by merging predictions after processing.
5. **Keep models as-is** (no retraining).

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Detect cores for parallelization
n_cores <- max(1, detectCores() - 1)

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

# Parallel prediction by year
years <- unique(pred_db$year)

cl <- makeCluster(n_cores)
clusterExport(cl, c("joined_data", "rf_models_per_year", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

pred_list <- parLapply(cl, years, function(yr) {
  if (as.character(yr) %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[as.character(yr)]]
    test_set <- joined_data[year == yr]
    preds <- predict_in_chunks(model, test_set)
    data.table(id = test_set$id, pred = preds) # assuming 'id' exists
  } else {
    NULL
  }
})

stopCluster(cl)

# Combine predictions and merge back
pred_dt <- rbindlist(pred_list, use.names = TRUE, fill = TRUE)
setkey(pred_db, id)
setkey(pred_dt, id)
pred_db[pred_dt, consolidated := i.pred]

fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Benefits:**  
- **Chunking** prevents memory overload.
- **Parallelization** speeds up processing across years.
- **`data.table`** ensures fast filtering and merging.
- Preserves trained models and avoids retraining.  

This approach should make the process feasible on a normal machine.