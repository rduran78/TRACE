 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. This is memory-intensive because:
- Entire `test_set` is held in memory for each year.
- `predict()` processes all rows at once.
- `joined_data` and `pred_db` are large, causing repeated subsetting and assignment.

**Optimization Strategy:**  
- Use **data.table** for efficient filtering and assignment instead of `dplyr`.
- Predict in **chunks/batches** to avoid loading all rows into memory at once.
- Avoid copying large objects repeatedly.
- Keep models in memory but process data sequentially in smaller blocks.
- Write intermediate results to disk if memory is tight.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Ensure pred_db has 'consolidated' column
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Define chunk size (adjust based on memory)
chunk_size <- 50000

# Loop over years
for (year in unique(pred_db$year)) {
  cat("Predicting for year", year, "\n")
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Subset rows for this year
    idx <- which(pred_db$year == year)
    test_set <- joined_data[year == !!year]
    
    # Predict in chunks
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      end_i <- min(i + chunk_size - 1, n)
      preds[i:end_i] <- predict(rf_model, newdata = test_set[i:end_i])
    }
    
    # Assign predictions back
    pred_db[idx, consolidated := preds]
    
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
  
  # Optional: write partial results to disk to save memory
  fwrite(pred_db, "RF_imputated_db_partial.csv")
}

# Final write
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` minimizes memory copies and speeds up filtering/assignment.
- Chunked prediction prevents loading massive data into memory at once.
- Intermediate writes ensure progress is saved and reduce risk of crashes.

**Extra Tip:** If still slow, consider:
- Running in parallel with `parallel::mclapply` for chunks.
- Using `ranger` package for faster prediction (if models can be converted).