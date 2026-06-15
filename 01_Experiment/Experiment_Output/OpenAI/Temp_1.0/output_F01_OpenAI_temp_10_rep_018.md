 **Diagnosis:**  
The main bottleneck is the **for-loop iterating over years** and filtering `joined_data` for each year combined with predictions on very large data frames. Each iteration copies and subsets large objects, which is memory and time intensive. Predicting one year at a time using `rf_models_per_year` is correct but inefficient.  

---

### **Optimization Strategy**
1. **Vectorize and Minimize Copying:**  
   - Avoid repeated subsetting with `dplyr` inside a loop. Instead, use `data.table` for fast filtering and in-place updates.
2. **Chunk-Based Prediction:**  
   - Predict in chunks (`split` or data.table batching) to avoid loading entire data into memory at once.
3. **Parallelization:**  
   - Use `parallel` or `foreach` with appropriate backend for predictions.
4. **Preserve Models:**  
   - Don't retrain; reuse `rf_models_per_year`.
5. **I/O Optimization:**  
   - Write output once at the end, not inside the loop.

---

### **Optimized Working Code**

```r
library(data.table)
library(parallel)

# Convert data frames to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Ensure rf_models_per_year exists from loaded RData
# Detect available cores
num_cores <- max(1, detectCores() - 1)

# Function for prediction per year
predict_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict(rf_model, newdata = test_set)
    return(data.table(cell_id = test_set$cell_id, year = year, preds = preds))
  } else {
    warning(paste0("No model found for year ", year))
    return(NULL)
  }
}

# Parallel apply over years
years <- unique(pred_db$year)
res_list <- mclapply(years, predict_year, mc.cores = num_cores)

# Combine all predictions
pred_all <- rbindlist(res_list, use.names = TRUE, fill = TRUE)

# Merge predictions back into pred_db by cell_id and year
setkey(pred_db, cell_id, year)
setkey(pred_all, cell_id, year)
pred_db[pred_all, consolidated := preds]

# Save results efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Why This is Better**
- `data.table` filtering and joins avoid repeated memory copying (much faster than `dplyr` for large data).
- `mclapply` runs predictions in parallel across available cores.
- Predicts in chunks per year without loading unnecessary data into memory.
- Final write uses `fwrite`, which is extremely fast for large files.

This approach should make the computation feasible on a standard machine while preserving the pre-trained models.