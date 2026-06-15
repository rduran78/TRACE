 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` on Random Forest models. This is memory-intensive because `joined_data` and `pred_db` are large, and the entire dataset is processed in one go without chunking or parallelization. Additionally, `dplyr` operations inside the loop can be slow for very large data.

---

**Optimization Strategy:**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.  
2. **Chunk predictions** into smaller batches to reduce memory load.  
3. **Parallelize predictions** using `parallel` or `future.apply` since Random Forest prediction is embarrassingly parallel.  
4. Avoid repeatedly copying large objects; work in-place with `data.table`.  
5. Preserve trained models as requested.

---

**Optimized R Code:**

```r
library(data.table)
library(parallel)

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Detect available cores for parallel processing
n_cores <- max(1, detectCores() - 1)

# Function to predict in chunks
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = newdata[idx[[i]], ])
  }
  return(preds)
}

# Parallel prediction by year
cl <- makeCluster(n_cores)
clusterExport(cl, c("rf_models_per_year", "joined_data", "predict_in_chunks"))
clusterEvalQ(cl, library(randomForest))

years <- unique(pred_db$year)

par_results <- parLapply(cl, years, function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict_in_chunks(rf_model, test_set)
    list(year = year, preds = preds, idx = which(pred_db$year == year))
  } else {
    NULL
  }
})

stopCluster(cl)

# Assign predictions back to pred_db
for (res in par_results) {
  if (!is.null(res)) {
    pred_db$consolidated[res$idx] <- res$preds
  }
}

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically speeds up filtering and assignment.  
- Chunking prevents memory overload during `predict()`.  
- Parallelization uses multiple cores to process years concurrently.  
- Preserves trained models without retraining.  

This approach should make the process feasible on a normal machine.