 **Diagnosis:**  
The bottleneck is likely due to applying `predict()` on large `test_set` objects sequentially for each year, with all data held in memory. This approach becomes memory-intensive and slow when handling hundreds of thousands of rows and many predictors at once.

---

**Optimization Strategy:**  
- **Reduce memory usage and improve speed** by:
  1. Using `data.table` for efficient filtering and assignment.
  2. Processing predictions in **chunks (batch processing)** instead of loading entire yearly data into memory at once.
  3. Avoiding repeated conversions (e.g., `dplyr` pipes) and leveraging base or `data.table` operations.
  4. Writing results incrementally to disk to prevent high memory usage.

---

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Ensure pred_db has a column for predictions
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Define chunk size (adjust based on available memory)
chunk_size <- 50000

# Loop over years
for (year in unique(pred_db$year)) {
  cat(sprintf("Predicting for year %s\n", year))
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Subset for the year
    test_set <- joined_data[year == ..year]
    
    # Process in chunks
    n <- nrow(test_set)
    idx_seq <- seq(1, n, by = chunk_size)
    
    for (start_idx in idx_seq) {
      end_idx <- min(start_idx + chunk_size - 1, n)
      chunk <- test_set[start_idx:end_idx]
      
      # Predict for chunk
      preds <- predict(rf_model, newdata = chunk)
      
      # Assign predictions back to pred_db
      pred_db[year == ..year][start_idx:end_idx, consolidated := preds]
    }
  } else {
    cat(sprintf("Warning: No model found for year %s\n", year))
  }
  
  # Optional: write intermediate results to disk
  fwrite(pred_db, "RF_imputated_db_partial.csv")
}

# Final save
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` significantly reduces overhead compared to `dplyr` for large datasets.
- Chunk processing prevents exhausting RAM when predicting large datasets.
- Incremental writing avoids keeping all predictions in memory until the end.

This approach should make the process feasible on a standard machine while preserving the trained Random Forest models.