 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` on Random Forest models. This is memory-intensive because `joined_data` and `pred_db` are large, and the entire dataset is processed in one go without chunking or parallelization.  

**Optimization Strategy:**  
- Use `data.table` for efficient filtering and assignment.  
- Process predictions in chunks to reduce memory load.  
- Optionally enable parallel prediction using `parallel::mclapply` or `future.apply`.  
- Avoid repeatedly copying large objects; work in-place.  

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Identify years
years <- unique(pred_db$year)

# Define chunk size (adjust based on memory)
chunk_size <- 50000

# Function to predict in chunks
predict_in_chunks <- function(model, data, chunk_size) {
  n <- nrow(data)
  preds <- vector("numeric", n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  return(preds)
}

# Parallel processing across years
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("joined_data", "pred_db", "rf_models_per_year", "predict_in_chunks", "chunk_size"))
clusterEvalQ(cl, library(randomForest))

parLapply(cl, years, function(year) {
  cat(paste0("Predicting for year ", year, "\n"))
  test_set <- joined_data[year == !!year]
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    preds <- predict_in_chunks(rf_model, test_set, chunk_size)
    pred_db[year == !!year, consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", year, "\n"))
  }
})

stopCluster(cl)

# Save results
fwrite(pred_db, "RF_imputated_db.csv")
```

**Key Improvements:**  
- `data.table` for fast filtering and assignment.  
- Chunked prediction to avoid memory overload.  
- Parallelization across years for speed.  

This approach preserves trained models and makes the process feasible on a normal machine.