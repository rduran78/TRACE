 **Diagnosis:**  
The bottleneck is the `for` loop predicting year by year on large datasets using `dplyr` filtering and `predict()` sequentially. This causes repeated subsetting and memory overhead. Since models are already trained, the main optimization is to **vectorize predictions**, reduce data copies, and use efficient data structures.

---

### **Optimization Strategy**
1. **Switch to `data.table`** for fast filtering and assignment in place.
2. **Avoid repeated filtering** by splitting data once by year.
3. **Parallelize predictions** using `parallel` or `future.apply`.
4. **Write output incrementally** to avoid holding everything in memory.
5. **Ensure predictors match model features** before prediction to prevent overhead.

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_split <- split(joined_data, joined_data$year)

# Detect available cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Parallel prediction
cl <- makeCluster(n_cores)
clusterExport(cl, c("rf_models_per_year", "data_split", "pred_db"))
clusterEvalQ(cl, library(randomForest))

pred_results <- parLapply(cl, names(data_split), function(yr) {
  if (yr %in% names(rf_models_per_year)) {
    model <- rf_models_per_year[[yr]]
    preds <- predict(model, newdata = data_split[[yr]])
    list(year = as.integer(yr), preds = preds, idx = which(pred_db$year == as.integer(yr)))
  } else {
    NULL
  }
})
stopCluster(cl)

# Assign predictions back efficiently
for (res in pred_results) {
  if (!is.null(res)) {
    pred_db$consolidated[res$idx] <- res$preds
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

✅ **Why this works:**  
- `data.table` minimizes memory copies.  
- `split()` avoids repeated filtering.  
- `parLapply()` distributes prediction tasks across cores.  
- Assigning by index avoids merging overhead.  
- Writing with `fwrite()` is faster than `write.csv()`.  

This approach should significantly reduce runtime and memory usage on a personal machine.