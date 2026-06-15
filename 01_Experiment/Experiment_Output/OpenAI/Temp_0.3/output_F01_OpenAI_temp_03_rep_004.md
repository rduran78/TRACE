 **Diagnosis:**  
The bottleneck is the loop performing `predict()` on large `test_set` subsets for each year, which is memory-intensive and slow because:
- `joined_data %>% filter(year == year)` repeatedly subsets a large dataset.
- Predictions are stored in `pred_db` row by row, which is inefficient.
- No parallelization or chunking is used for large data.

---

**Optimization Strategy:**  
1. **Use `data.table` for fast filtering and assignment** instead of `dplyr`.
2. **Chunk predictions** for large subsets to reduce memory load.
3. **Avoid growing objects repeatedly**; preallocate and update efficiently.
4. **Leverage parallel processing** with `parallel::mclapply` or `future.apply`.
5. **Keep models intact** (no retraining).

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Identify unique years
years <- unique(pred_db$year)

# Detect number of cores
n_cores <- max(1, detectCores() - 1)

# Function to predict for a single year in chunks
predict_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]

    # Chunking
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    return(data.table(id = test_set$id, pred = preds))
  } else {
    warning(paste("No model for year", year))
    return(NULL)
  }
}

# Parallel prediction
results <- mclapply(years, predict_year, mc.cores = n_cores)
results <- rbindlist(results, use.names = TRUE, fill = TRUE)

# Merge predictions back
setkey(pred_db, id)
setkey(results, id)
pred_db[results, consolidated := i.pred]

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**
- `data.table` drastically speeds up filtering and merging.
- Chunking prevents memory overload.
- Parallelization accelerates processing across years.
- Preserves trained models and avoids retraining.