 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` on full data frames. This is memory-intensive because `joined_data` and `pred_db` are large, and the entire prediction is done in one go per year. Additionally, `dplyr` operations and base R indexing on large objects are slow.

---

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment instead of `dplyr`.
- Process predictions in **chunks** to reduce memory load.
- Avoid repeatedly copying large objects; work in-place.
- Use `predict(..., newdata = chunk)` in batches rather than the entire year at once.
- Keep trained models intact (no retraining).

---

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure rf_models_per_year is loaded
# pred_db has columns: year, consolidated
# joined_data has predictors and year

# Chunk size (adjust based on memory)
chunk_size <- 50000

years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter test set for this year
    test_set <- joined_data[year == yr]
    
    # Get row indices in pred_db for assignment
    idx <- which(pred_db$year == yr)
    
    # Process in chunks
    n <- nrow(test_set)
    for (start in seq(1, n, by = chunk_size)) {
      end <- min(start + chunk_size - 1, n)
      chunk <- test_set[start:end]
      
      # Predict on chunk
      preds <- predict(rf_model, newdata = chunk)
      
      # Assign predictions
      pred_db$consolidated[idx[start:end]] <- preds
    }
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically reduces filtering and assignment overhead.
- Chunking prevents memory overload during prediction.
- Preserves trained models and avoids retraining.
- Scales better for hundreds of thousands of rows on a personal computer.