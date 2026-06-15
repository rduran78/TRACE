 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large data frames with `dplyr` filtering and base R assignment. Both operations are slow and memory-heavy for hundreds of thousands of rows per year. Predictions themselves are fast, but repetitive subsetting and copying of large data frames cause inefficiency.

---

**Optimization Strategy:**  
1. Avoid repeated `dplyr` filtering and base assignment inside the loop—switch to `data.table` for in-place operations.
2. Pre-split or index by year to reduce overhead.
3. Use `predict(..., newdata, type="response")` in vectorized chunks if memory is limited.
4. Ensure factor levels in `test_set` match the model (already assumed).
5. Write output incrementally or after processing to avoid memory duplication.

---

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table for efficient filtering and assignment
setDT(pred_db)
setDT(joined_data)

# Create an index for faster subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Loop over models efficiently
for (yr in names(rf_models_per_year)) {
  cat("Predicting for year", yr, "\n")
  
  # Subset using data.table (no copy)
  test_set <- joined_data[J(as.integer(yr))]  # or as.numeric if needed
  
  if (nrow(test_set) > 0) {
    rf_model <- rf_models_per_year[[yr]]
    
    # Predict in chunks to reduce memory load
    chunk_size <- 50000
    preds <- vector("numeric", nrow(test_set))
    
    for (i in seq(1, nrow(test_set), by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, nrow(test_set))
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions in place
    pred_db[J(as.integer(yr)), consolidated := preds]
  } else {
    cat("Warning: No data for year", yr, "\n")
  }
}

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this helps:**  
- `data.table` avoids repeated copies and is far faster for filtering and assignment.
- Chunked prediction prevents exhausting RAM when `newdata` is large.
- Keys (`setkey`) make joins and lookups O(1) per group.
- Preserves trained models and only performs predictions.  

This approach should make the process feasible on a personal computer.