 **Diagnosis:**  
The bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and repeated subsetting. This approach is memory-inefficient and slow because:  
- `filter()` and `%>%` create copies of large data frames repeatedly.  
- Predictions are done year by year instead of in a vectorized or chunked manner.  
- `pred_db$consolidated[...] <- ...` repeatedly modifies a large object in memory.  

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment (avoids repeated copies).  
- Predict in chunks to reduce memory load.  
- Avoid unnecessary joins or conversions inside the loop.  
- Keep models as is (no retraining).  
- Write results incrementally or after processing all years.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure 'year' is keyed for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Add consolidated column if not present
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Function to predict in chunks
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  for (i in seq(1, n, by = chunk_size)) {
    idx <- i:min(i + chunk_size - 1, n)
    preds[idx] <- predict(model, newdata = newdata[idx])
  }
  preds
}

# Loop over years efficiently
for (yr in unique(pred_db$year)) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset data for this year
    test_set <- joined_data[year == yr]
    
    if (nrow(test_set) > 0) {
      preds <- predict_in_chunks(rf_model, test_set)
      
      # Assign predictions back efficiently
      pred_db[year == yr, consolidated := preds]
    }
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` subsetting and assignment are in-place (no large copies).  
- Chunked prediction prevents memory overload.  
- Preserves trained models and original logic.  
- Scales better for hundreds of thousands of rows.  

**Extra tip:** If memory is still tight, process and write each year’s predictions to disk immediately instead of holding everything in memory.