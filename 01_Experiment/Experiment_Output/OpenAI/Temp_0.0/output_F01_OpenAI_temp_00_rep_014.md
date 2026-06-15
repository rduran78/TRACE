 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated indexing. This approach is memory-inefficient and slow because:
- `filter()` and subsetting inside the loop repeatedly copy large data frames.
- Predictions are done year by year instead of in a vectorized or chunked manner.
- `pred_db$consolidated[...] <- ...` triggers repeated writes to a large object.

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment (avoids repeated copies).
- Predict in **chunks** to reduce memory load.
- Avoid creating intermediate large objects unnecessarily.
- Keep models as is (no retraining).
- Write results incrementally or after processing all chunks.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure 'year' is keyed for fast filtering
setkey(pred_db, year)
setkey(joined_data, year)

# Function to predict in chunks
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = newdata[idx[[i]], ])
  }
  preds
}

# Main prediction loop
for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter efficiently
    test_set <- joined_data[year == yr]
    
    # Predict in chunks
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign back efficiently
    pred_db[year == yr, consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically reduces memory copies and speeds up filtering/assignment.
- Chunked prediction prevents memory overload when `predict()` handles large data.
- Preserves trained models and original logic, but makes it feasible on a normal machine.  

**Extra tip:** If memory is still tight, process and write **year by year** to disk instead of holding the full `pred_db` in memory.