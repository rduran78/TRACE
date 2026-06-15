 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated indexing. This approach is memory-inefficient and slow because:
- `filter()` and subsetting inside the loop repeatedly copy large data frames.
- Predictions are done year by year instead of in a vectorized or batched manner.
- `pred_db$consolidated[...] <- ...` triggers repeated writes to a large object.

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment (avoids repeated copies).
- Pre-split data by year once, or use `data.table` grouping.
- Predict in chunks if memory is limited (batch processing).
- Avoid growing objects inside the loop.
- Keep models as is (no retraining).

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure 'year' is the same type in both
pred_db[, year := as.character(year)]
joined_data[, year := as.character(year)]

# Initialize consolidated column
pred_db[, consolidated := NA_real_]

# Process by year using data.table grouping
for (yr in names(rf_models_per_year)) {
  cat("Predicting for year", yr, "\n")
  
  rf_model <- rf_models_per_year[[yr]]
  
  # Subset joined_data for this year
  test_set <- joined_data[year == yr]
  
  if (nrow(test_set) > 0) {
    # Predict in chunks to avoid memory overload
    chunk_size <- 50000
    preds <- numeric(nrow(test_set))
    
    for (i in seq(1, nrow(test_set), by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, nrow(test_set))
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions back to pred_db efficiently
    pred_db[year == yr, consolidated := preds]
  } else {
    cat("Warning: No data for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` avoids repeated copies and speeds up filtering/assignment.
- Chunked prediction prevents memory overload.
- Preserves trained models and original logic.
- Scales better for hundreds of thousands of rows.

**Extra tip:** If `joined_data` and `pred_db` share a unique ID, merge predictions back by ID instead of relying on row order for robustness.