 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` on Random Forest models. This is memory-intensive because `joined_data` and `pred_db` are large, and the entire dataset is processed in one go without chunking or parallelization.

---

**Optimization Strategy:**  
- Use **data.table** for efficient filtering and assignment.
- Process predictions in **chunks** to reduce memory load.
- Enable **parallel processing** for predictions using `parallel::mclapply` or `future.apply`.
- Avoid repeatedly copying large objects; work with references.
- Write intermediate results to disk if memory is tight.

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Detect available cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Function to predict for one year in chunks
predict_year <- function(year) {
  rf_model <- rf_models_per_year[[as.character(year)]]
  if (is.null(rf_model)) {
    message("Warning: No model for year ", year)
    return(NULL)
  }
  
  test_set <- joined_data[year == !!year]
  
  # Chunking
  chunk_size <- 50000
  n <- nrow(test_set)
  preds <- vector("numeric", n)
  
  for (i in seq(1, n, by = chunk_size)) {
    end_i <- min(i + chunk_size - 1, n)
    preds[i:end_i] <- predict(rf_model, newdata = test_set[i:end_i])
  }
  
  list(year = year, preds = preds, idx = pred_db[year == !!year, .I])
}

# Parallel prediction
results <- mclapply(unique(pred_db$year), predict_year, mc.cores = n_cores)

# Assign predictions back
for (res in results) {
  if (!is.null(res)) {
    pred_db[res$idx, consolidated := res$preds]
  }
}

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` reduces memory overhead and speeds up filtering.
- Chunking prevents loading huge subsets into memory at once.
- Parallelization uses multiple cores for faster execution.
- Preserves trained models without retraining.  

This approach makes the process feasible on a normal machine while handling large datasets efficiently.