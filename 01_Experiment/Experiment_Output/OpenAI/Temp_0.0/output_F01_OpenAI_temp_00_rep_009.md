 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` on Random Forest models. This is memory-intensive because `joined_data` and `pred_db` are large, and the entire dataset is processed in one go without chunking or parallelization.

---

**Optimization Strategy:**  
- Use **data.table** for efficient filtering and assignment.
- Apply **chunked/batched prediction** to avoid loading huge subsets into memory at once.
- Use **parallel processing** with `parallel::mclapply` or `future.apply` for multi-core prediction.
- Avoid repeated copying of large objects; work in-place.
- Preserve trained models as requested.

---

**Optimized R Code:**

```r
library(data.table)
library(parallel)

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Get unique years
years <- unique(pred_db$year)

# Define prediction function with chunking
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  return(preds)
}

# Parallel prediction across years
num_cores <- max(1, detectCores() - 1)
mclapply(years, function(year) {
  cat(paste0("Predicting for year ", year, "\n"))
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict_in_chunks(rf_model, test_set)
    pred_db[year == !!year, consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", year, "\n"))
  }
}, mc.cores = num_cores)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**  
- `data.table` reduces memory overhead and speeds up filtering/assignment.
- Chunking (`chunk_size`) prevents memory overload during prediction.
- `mclapply` uses multiple cores for parallel year-level processing.
- No retraining; models are reused as requested.

This approach makes the process feasible on a normal machine by reducing memory footprint and leveraging parallelism.