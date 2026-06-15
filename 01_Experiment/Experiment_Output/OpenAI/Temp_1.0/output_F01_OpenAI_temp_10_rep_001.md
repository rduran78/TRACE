 **Diagnosis:**  
The bottlenecks likely come from (a) looping over years at the R level, (b) using `dplyr` filtering repeatedly on large data, and (c) predicting row by row instead of in chunks. The Random Forest `predict()` call can handle large batches, but memory blows up when copying big data frames for each year.  

**Optimization Strategy:**  
- Avoid repeated `dplyr` filtering; use `data.table` for fast subsetting and in-place assignment.  
- Process predictions in chunks to reduce memory footprint.  
- Use `rbindlist` and preallocate result columns.  
- Keep models unchanged (already trained).  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Preallocate consolidated column
pred_db[, consolidated := NA_real_]

# Optional: Set keys for fast joins
setkey(pred_db, year)
setkey(joined_data, year)

# Loop over years
for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")

  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]

    # Extract rows for this year efficiently
    rows_idx <- which(pred_db$year == yr)
    test_set <- joined_data[year == yr]

    # Predict in chunks to avoid memory overload
    chunk_size <- 50000
    preds <- numeric(nrow(test_set))
    for (i in seq(1, nrow(test_set), by = chunk_size)) {
      end_i <- min(i + chunk_size - 1, nrow(test_set))
      preds[i:end_i] <- predict(rf_model, newdata = test_set[i:end_i, ])
    }

    # Assign predictions back
    pred_db[rows_idx, consolidated := preds]
  } else {
    cat("Warning: No model for year", yr, "\n")
  }
}

# Write result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this is faster & feasible:**  
- `data.table` operations are in-place → less copying.  
- Chunked prediction prevents RAM spikes.  
- Loop overhead reduced significantly.  

This should run on a normal machine for hundreds of thousands of rows while preserving your trained models.