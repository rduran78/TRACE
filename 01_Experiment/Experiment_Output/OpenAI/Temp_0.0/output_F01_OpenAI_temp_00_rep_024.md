 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated indexing. This approach is memory-inefficient and slow because:
- `filter()` and subsetting inside the loop repeatedly copy large data frames.
- Predictions are done year by year instead of in a vectorized or chunked manner.
- `pred_db$consolidated[...] <- ...` triggers repeated writes to a large object.

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment (avoids repeated copies).
- Process data in chunks or by year but without copying entire data frames.
- Pre-allocate the prediction column and update by reference.
- Avoid unnecessary conversions and keep everything in memory-efficient structures.
- Use `predict(..., newdata, type="response")` in batch mode for each year.

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Pre-allocate prediction column
pred_db[, consolidated := NA_real_]

# Loop through years efficiently
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter joined_data for this year without copying
    test_idx <- joined_data$year == yr
    test_set <- joined_data[test_idx]
    
    # Predict in batch
    preds <- predict(rf_model, newdata = test_set)
    
    # Assign predictions by reference
    pred_db[year == yr, consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` avoids repeated copies and uses in-place updates.
- Filtering and assignment are O(1) operations by reference.
- Predictions are still done per year (to match model availability) but without overhead from `dplyr`.
- Memory footprint is minimized, making it feasible on a normal machine.

**Extra tip:** If memory is still an issue, process and write results year by year (streaming) instead of holding the entire `pred_db` in memory.