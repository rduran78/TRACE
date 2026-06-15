 **Diagnosis:**  
The bottleneck is the loop performing predictions year by year on large datasets using `predict()` on Random Forest models. This is memory-intensive because `joined_data` is large, and `predict()` processes all rows at once. Reading and writing large objects also adds overhead.

**Optimization Strategy:**  
- Use `data.table` for efficient filtering and assignment instead of `dplyr`.
- Process predictions in chunks to reduce memory load.
- Avoid repeatedly copying large objects; work with references.
- Use `fwrite()` for faster output.
- Keep models intact as required.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Ensure rf_models_per_year is loaded
# load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Initialize consolidated column if not present
if (!"consolidated" %in% names(pred_db)) {
  pred_db[, consolidated := NA_real_]
}

# Define chunk size (adjust based on available RAM)
chunk_size <- 50000

years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Filter test set for the year
    test_set <- joined_data[year == yr]
    
    # Predict in chunks
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      end_i <- min(i + chunk_size - 1, n)
      preds[i:end_i] <- predict(rf_model, newdata = test_set[i:end_i])
    }
    
    # Assign predictions efficiently
    pred_db[year == yr, consolidated := preds]
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Fast write to CSV
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `data.table` filtering and assignment avoids expensive copies.
- Chunked prediction prevents memory overload.
- `fwrite()` is faster than `write.csv()`.
- Preserves trained models without retraining.

This approach should make the process feasible on a normal machine while handling large datasets efficiently.