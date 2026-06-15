 **Diagnosis:**  
The bottleneck is the `for` loop predicting year by year on large data frames using `dplyr::filter` repeatedly, which is slow and memory-heavy. Additionally, `predict()` is called on full-year subsets, which can still be huge. Key inefficiencies:  
- Repeated filtering with `%>% filter(year == ...)`.
- Non-vectorized assignment to `pred_db$consolidated`.
- Lack of chunking for large prediction sets.
- Using `dplyr` in a loop instead of more efficient tools like `data.table`.

---

**Optimization Strategy:**  
1. Convert `pred_db` and `joined_data` to `data.table` for fast filtering and assignment.  
2. Avoid repeated filtering by splitting data once or leveraging `by` operations.  
3. Predict in **chunks** to reduce memory usage.  
4. Use `set()` for fast assignment instead of `$`.  
5. Preserve trained models (no retraining).

---

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# Ensure matching keys
setkey(pred_db, year)
setkey(joined_data, year)

# Chunk size for predictions (adjust based on RAM)
chunk_size <- 50000

for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset data for this year
    test_set <- joined_data[year == yr]
    
    if (nrow(test_set) > 0) {
      # Predict in chunks
      pred_values <- vector("numeric", nrow(test_set))
      n_chunks <- ceiling(nrow(test_set) / chunk_size)
      
      for (i in seq_len(n_chunks)) {
        idx_start <- (i - 1) * chunk_size + 1
        idx_end <- min(i * chunk_size, nrow(test_set))
        
        chunk <- test_set[idx_start:idx_end]
        pred_values[idx_start:idx_end] <- predict(rf_model, newdata = chunk)
      }
      
      # Assign predictions back efficiently
      pred_db[year == yr, consolidated := pred_values]
    }
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why it’s faster and lighter:**  
- `data.table` filtering and assignment avoid expensive copy operations.  
- Predictions are done in memory-friendly **chunks**, preventing crashes.  
- Minimal overhead inside the loop.  

**Extra tip:** If the machine still struggles, consider:  
- Reducing `chunk_size` further.  
- Running on multiple cores using `parallel::mclapply` over years or chunks.  
- Saving intermediate results to disk for very large datasets.