 **Diagnosis:**  
- The loop processes predictions year by year on the full dataset, which may include creating large intermediate `test_set` objects.
- `dplyr` operations (`filter`) on huge data frames in every iteration add overhead.
- RandomForest `predict()` is relatively fast, but the data preparation step is slowing things down and using too much memory.
- No parallelization or optimized data structures is being used, and the entire `pred_db` is kept in memory.

---

### **Optimization Strategy**
1. **Switch to `data.table` fully:** Use efficient subsetting and assignment to avoid repeated copies.
2. **Chunk or stream data:** Predict on smaller chunks instead of holding everything in memory.
3. **Parallel predictions:** If supported by RandomForest predict, leverage `parallel::mclapply` or `future.apply`.
4. **Avoid constructing large temp objects:** Use column assignment by reference.
5. **Save incrementally:** Write out predictions periodically to reduce memory footprint.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# Ensure rf_models_per_year is loaded and named by year
years <- unique(pred_db$year)

# Create consolidated column efficiently
pred_db[, consolidated := NA_real_]

# Process in chunks with data.table, no copying
for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Extract only rows for this year directly as data.table
    idx <- which(pred_db$year == yr)
    test_set <- joined_data[year == yr]
    
    # Predict in smaller batches to avoid memory overload
    batch_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (start in seq(1, n, by = batch_size)) {
      end <- min(start + batch_size - 1, n)
      preds[start:end] <- predict(rf_model, newdata = test_set[start:end])
    }
    
    # Assign by reference
    pred_db[idx, consolidated := preds]
  } else {
    cat("Warning: No model for year", yr, "\n")
  }
  
  # Optional: write partial results to disk for memory safety
  fwrite(pred_db, file = "RF_imputated_db_partial.csv")
}

# Final save
fwrite(pred_db, "RF_imputated_db.csv")
```

---

#### **Why This Works**
- `data.table` optimizes subsetting and assignment (reduces memory use & speed).
- Processing in **chunks** prevents blowing up RAM.
- `fwrite()` saves large data efficiently and can be used periodically to avoid risk of data loss.
- Preserves existing trained models without retraining.

This approach makes predictions computationally feasible on a normal machine by reducing RAM overhead and improving I/O speed.