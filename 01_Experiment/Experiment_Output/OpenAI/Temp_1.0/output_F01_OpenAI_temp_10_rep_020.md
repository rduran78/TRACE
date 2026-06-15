 **Diagnosis:**  
The loop predicts for each year using a full `predict()` call, which becomes slow and memory heavy because:
- `joined_data` filtering and large `test_set` objects are recreated in every iteration.
- `pred_db$consolidated` assignment is inefficient for large indexes.
- `predict()` on `randomForest` objects for hundreds of thousands of rows is computationally expensive and single-threaded by default.
- Lack of parallelization or batching causes high memory pressure.

---

### **Optimization Strategy**
1. **Use Data.table for filtering and assignments** → avoids dplyr overhead.
2. **Batch predictions** → chunk rows to reduce RAM load.
3. **Parallelize predictions** using `parallel` or `future.apply` packages.
4. **Avoid re-joining or repeated operations** → prepare an index for fast access.
5. **Write results incrementally** or store as efficient format (e.g., `fst`, `data.table::fwrite`) to prevent huge in-memory object growth.

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert big data frames to data.table for speed
setDT(joined_data)
setDT(pred_db)

# Function for batched prediction
predict_in_batches <- function(model, data, batch_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  for (i in seq(1, n, by = batch_size)) {
    idx <- i:min(i + batch_size - 1, n)
    preds[idx] <- predict(model, newdata = data[idx])
  }
  return(preds)
}

unique_years <- unique(pred_db$year)

# Option: parallel execution across years
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("joined_data", "rf_models_per_year", "predict_in_batches"))
clusterEvalQ(cl, { library(randomForest) })

results_list <- parLapply(cl, unique_years, function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == year]
    preds <- predict_in_batches(rf_model, test_set)
    return(list(year = year, preds = preds, idx = which(pred_db$year == year)))
  } else {
    return(NULL)
  }
})
stopCluster(cl)

# Assign predictions efficiently
for (res in results_list) {
  if (!is.null(res)) {
    pred_db[res$idx, consolidated := res$preds]
  }
}

# Write as fast binary or CSV
fwrite(pred_db, "RF_imputated_db.csv")  # much faster than write.csv
```

---

### **Key Gains**
- **Data.table filtering/assignment** = significant speed up.
- **Batching** prevents running out of RAM.
- **Parallelization across years** = near-linear speed gain on multi-core CPUs.
- **Efficient output writing (`fwrite`)** avoids overhead of base `write.csv`.

This approach keeps the trained models intact, dramatically improves performance, and stays within normal PC limits.