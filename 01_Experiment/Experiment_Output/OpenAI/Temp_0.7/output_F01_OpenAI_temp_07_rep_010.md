 **Diagnosis**  
- The bottleneck is the loop over years with repeated filtering (`joined_data %>% filter(year == year)`), creating large intermediate objects.
- `predict()` on very large `data.frame`s can be memory-intensive because `randomForest` copies data internally.
- All predictions are being stored in `pred_db$consolidated`, which may be huge, stressing RAM.
- Using `dplyr` for filtering inside the loop introduces extra overhead for large datasets.

---

**Optimization Strategy**  
1. Use `data.table` for efficient subsetting and assignment (avoids repeated copies).
2. Process predictions in **chunks** (e.g., by rows) to reduce memory load.
3. Use `predict(..., newdata = , type="response")` directly on `data.table` subsets without converting back to `data.frame`.
4. Write results incrementally to disk instead of keeping everything in memory.
5. Avoid unnecessary objects; reuse column references rather than duplicating.

---

**Optimized R Code**  
```r
library(data.table)

# Convert joined_data and pred_db to data.table
setDT(joined_data)
setDT(pred_db)

# Ensure 'year' is keyed for fast filtering
setkey(joined_data, year)
setkey(pred_db, year)

# Create output file and write header
fwrite(pred_db[0], "RF_imputated_db.csv")  # empty structure for header

# Process in chunks by year
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Extract subset for this year
    test_set <- joined_data[J(yr)]
    
    # Predict in chunks to avoid memory blow-up
    chunk_size <- 50000  # adjust as needed
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions
    pred_db[J(yr), consolidated := preds]
    
    # Write incremental results to CSV
    fwrite(pred_db[J(yr)], "RF_imputated_db.csv", append = TRUE)
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Optionally remove in-memory pred_db to save RAM
rm(pred_db)
gc()
```

---

**Key Gains**  
- `data.table` reduces filtering overhead drastically.
- Chunking predictions prevents exhausting RAM.
- Incremental writing avoids holding full prediction matrix in memory.
- Preserves trained Random Forest models (no retraining).

This approach makes the process feasible on a standard computer with large datasets.