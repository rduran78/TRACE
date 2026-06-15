 **Diagnosis:**  
The main bottleneck is the loop predicting year by year on large datasets (`joined_data` and `pred_db`) using `dplyr` filtering and base R indexing. This approach repeatedly subsets large data frames and writes predictions back, which is memory- and time-intensive.  

**Optimization Strategy:**  
- Use `data.table` for efficient filtering and assignment instead of `dplyr` and base R.
- Avoid repeated subsetting of large objects inside the loop.
- Pre-split data by year once, then process in chunks.
- Use `predict(..., newdata, type="response")` in a vectorized way.
- Optionally parallelize predictions if multiple cores are available.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Ensure 'year' is keyed for fast joins
setkey(pred_db, year)
setkey(joined_data, year)

# Pre-split joined_data by year to avoid repeated filtering
joined_list <- split(joined_data, by = "year")

# Initialize consolidated column
pred_db[, consolidated := NA_real_]

# Loop through available models
for (yr in names(rf_models_per_year)) {
  cat("Predicting for year", yr, "\n")
  
  if (yr %in% names(joined_list)) {
    rf_model <- rf_models_per_year[[yr]]
    test_set <- joined_list[[yr]]
    
    # Predict in chunks to manage memory
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions back efficiently
    pred_db[year == as.integer(yr), consolidated := preds]
  } else {
    cat("Warning: No data found for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` drastically reduces memory overhead and speeds up filtering/assignment.
- Pre-splitting avoids repeated filtering inside the loop.
- Chunked prediction prevents memory overload when handling hundreds of thousands of rows.
- Preserves trained models as required.  

Optional: Add `parallel::mclapply` for multi-core prediction if needed.