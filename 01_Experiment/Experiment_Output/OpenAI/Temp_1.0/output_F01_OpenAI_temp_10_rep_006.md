 **Diagnosis:**  
The process is slow due to three factors:  
1. Iterating with `for` loops and repeated filtering using `%>% filter()` on large datasets is inefficient.  
2. `predict()` on large test sets for Random Forest is memory-intensive.  
3. Using `dplyr` with a huge dataset could cause unnecessary memory copies.  

**Optimization Strategy:**  
- Use `data.table` for fast subsetting and assignment (avoids full copies).  
- Process predictions in **chunks** (batching rows into smaller groups) to reduce memory strain.  
- Avoid creating temporary large objects repeatedly.  
- Preserve trained models as is (per requirement).  

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert pred_db and joined_data to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Ensure rf_models_per_year is already loaded
# Add consolidated column if not present
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Batch prediction function
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  seq_chunks <- seq(1, n, by = chunk_size)
  
  for (i in seq_along(seq_chunks)) {
    start <- seq_chunks[i]
    end <- min(start + chunk_size - 1, n)
    preds[start:end] <- predict(model, newdata[start:end])
  }
  return(preds)
}

unique_years <- unique(pred_db$year)

for (year in unique_years) {
  cat(sprintf("Predicting for year %s\n", year))
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]

    # Subset using data.table (very fast)
    test_set <- joined_data[year == !!year]
    idx <- pred_db[year == !!year, which = TRUE]  # indices for assignment

    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000)
    pred_db[idx, consolidated := preds]
  } else {
    cat(sprintf("Warning: No model found for year %s\n", year))
  }
}

fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Key Improvements**
- **`data.table` subsetting and `fwrite()`** → faster and memory-efficient.
- **Batch prediction (`predict_in_chunks`)** → prevents memory overload.
- **Avoids `%>%` and repeated `filter()`**, which are costly on big datasets.
- **Single consolidated object in memory**.

This approach will massively reduce run time and memory use while preserving the trained Random Forest models.