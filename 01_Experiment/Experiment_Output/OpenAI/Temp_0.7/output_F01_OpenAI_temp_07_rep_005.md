 **Diagnosis:**  
The current bottleneck is the loop predicting year-by-year on large data frames using standard `predict()` with `randomForest`. This is memory-intensive because:
- `joined_data` likely holds all years, but you filter repeatedly.
- Predictions happen on large chunks without batching.
- Results are written only after the entire loop, risking RAM overload.
- `tidyverse` and `sf` operations are slower for very large datasets.

---

### **Optimization Strategy**
- Use `data.table` for fast filtering and assignment.
- Process predictions in **batches** to avoid loading the entire dataset into RAM.
- Avoid copying large objects repeatedly.
- Write results incrementally or in chunks to disk (streaming).
- Keep models in memory but avoid retraining (as required).
- Use `predict(..., type = "response")` with `randomForest` efficiently.
- Drop unused columns from `test_set` to minimize overhead.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Open a connection for incremental writing
fwrite(pred_db[0], "RF_imputated_db.csv")  # write header only

# Batch size (adjust based on memory)
batch_size <- 50000

for (year in unique(pred_db$year)) {
  cat(sprintf("Predicting for year %s\n", year))
  
  # Check if model exists
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Extract rows for this year
    year_rows <- pred_db$year == year
    test_set <- joined_data[year == !!year]
    
    # Drop non-predictor columns (keep only model variables)
    predictors <- rf_model$forest$xlevels |> names()
    test_set <- test_set[, ..predictors]
    
    # Process in chunks
    idx <- which(year_rows)
    n <- length(idx)
    for (start in seq(1, n, by = batch_size)) {
      end <- min(start + batch_size - 1, n)
      rows <- idx[start:end]
      
      preds <- predict(rf_model, newdata = test_set[start:end])
      pred_db$consolidated[rows] <- preds
    }
    
  } else {
    cat(sprintf("Warning: No model found for year %s\n", year))
  }
  
  # Incremental write to CSV after each year
  fwrite(pred_db[year == !!year], "RF_imputated_db.csv", append = TRUE)
}

cat("Prediction completed and saved to RF_imputated_db.csv\n")
```

---

### **Why This Works**
- `data.table` drastically speeds up filtering and assignment.
- Chunked prediction prevents memory overload.
- Writing incrementally avoids holding the entire output in RAM.
- Model objects remain unchanged (preserved as requested).

**Extra Tip:** If prediction still feels slow, consider enabling **parallel prediction** using `parallel::mclapply` or the `foreach` + `doParallel` framework for each batch.