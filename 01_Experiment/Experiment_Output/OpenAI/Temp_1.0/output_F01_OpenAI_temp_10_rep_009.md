 **Diagnosis**  
The main performance bottleneck is the per-year loop applying `predict()` to large data subsets using base data frames and `%>%`. This approach:  
- Loads all data into memory and repeatedly filters it.  
- Uses standard R data frames which are slow for large numeric operations.  
- Does predictions sequentially without leveraging efficient methods for vectorized or chunked operations.  

Since the Random Forest models are already trained, the heavy lifting is strictly inference. Prediction on hundreds of thousands of rows at once can overwhelm memory.  

---

**Optimization Strategy**  
1. **Use `data.table` for fast filtering and assignment** rather than `%>%`.  
2. **Chunk predictions** for large subsets to avoid exhausting RAM.  
3. **Convert prediction data to `data.frame`** first (where required by `predict()`), but avoid unnecessary copies.  
4. **Pre-split data by year indices** instead of repeatedly filtering using dplyr.  
5. **Write output incrementally** to disk if memory is tight.  

---

**Optimized R Code**  
```r
library(data.table)

# Load prediction database and models
pred_db <- as.data.table(pred_db)
joined_data <- as.data.table(joined_data)

# Create a column for predictions
pred_db[, consolidated := NA_real_]

# Function for chunked prediction
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  out <- numeric(n)
  seq_starts <- seq(1, n, by = chunk_size)
  for (start in seq_starts) {
    end <- min(start + chunk_size - 1, n)
    chunk <- newdata[start:end]
    out[start:end] <- predict(model, newdata = as.data.frame(chunk))
  }
  out
}

# Precompute year grouping
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Get row indices once
    idx_pred <- which(pred_db$year == yr)
    idx_data <- which(joined_data$year == yr)
    
    if (length(idx_data) > 0) {
      test_set <- joined_data[idx_data]
      preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000)
      pred_db$consolidated[idx_pred] <- preds
    }
  } else {
    cat("Warning: No model for year", yr, "\n")
  }
  
  # Optional: write partial results to disk to save memory
  # fwrite(pred_db[idx_pred], sprintf("predictions_year_%s.csv", yr))
}

# Write final output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why This Works**  
- `data.table` drastically reduces memory overhead and filtering time.  
- Chunked prediction prevents massive data frames and memory spikes during `predict()`.  
- Minimal copying of data and avoids repeatedly invoking `%>%` pipelines.  

This approach keeps existing trained models intact and makes the workflow feasible on a single machine without requiring more RAM or retraining.