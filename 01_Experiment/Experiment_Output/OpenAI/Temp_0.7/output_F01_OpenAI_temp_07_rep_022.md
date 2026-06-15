 **Diagnosis:**  
The main performance bottlenecks are:  
1. The `for` loop iterates over each year with `dplyr::filter()`, which is slow for large datasets.  
2. Predictions are done row-wise with repeated subsetting of `pred_db`, adding overhead.  
3. Using base `predict()` on large data in one go may cause memory issues.  
4. Data is likely in `data.frame`/`tibble` format; converting to `data.table` and vectorized operations can speed up processing.  

---

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment.  
- Perform predictions in chunks (batch processing) to reduce memory load.  
- Avoid repeated filtering by splitting data by year once and processing in a loop.  
- Pre-allocate and write incrementally if needed.  
- Preserve the trained models as requested.  

---

**Optimized R Code:**  
```r
library(data.table)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure key for fast joins
setkey(pred_db, year)

# Function for chunked prediction
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx_seq <- split(seq_len(n), ceiling(seq_len(n)/chunk_size))
  for (idx in idx_seq) {
    preds[idx] <- predict(model, newdata = data[idx])
  }
  preds
}

# Initialize consolidated column
pred_db[, consolidated := NA_real_]

# Split joined_data by year for efficient access
data_by_year <- split(joined_data, by = "year")

# Iterate over years
for (yr in names(data_by_year)) {
  cat("Predicting for year", yr, "\n")
  
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    test_set <- data_by_year[[yr]]
    
    # Predict in chunks to manage memory
    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000)
    
    # Assign predictions back efficiently
    pred_db[year == as.integer(yr), consolidated := preds]
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` operations replace slow `dplyr::filter()` and subsetting.  
- Chunked prediction prevents memory overflow.  
- Avoids repeatedly scanning large data frames.  
- Maintains model integrity (no retraining).  

This approach should significantly improve speed and reduce memory usage on a normal machine.