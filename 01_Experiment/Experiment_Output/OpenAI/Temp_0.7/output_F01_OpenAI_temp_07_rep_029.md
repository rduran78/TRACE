 **Diagnosis**  
The bottleneck is caused by:  
- Iterating year-by-year with in-memory filtering of large `data.frame`s (`joined_data %>% filter(...)`), which is slow for hundreds of thousands of rows.  
- Performing `predict()` repeatedly on small subsets instead of in bulk.  
- `pred_db$consolidated` being updated element-wise, which is inefficient for large data.  
- Using `dplyr` pipelines and base loops together, which adds overhead.  

**Optimization Strategy**  
- Use `data.table` for efficient data handling and filtering.  
- Avoid repeated writes to large vectors inside loops; instead, compute predictions in a single table and then join.  
- Perform prediction in chunks to avoid memory overload but minimize overhead.  
- Keep trained models unchanged.  
- Use `fwrite()` for faster output writing.  

---

### **Optimized R Code**

```r
library(data.table)

# Load data as data.table
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for efficiency
setDT(pred_db)
setDT(joined_data)

# Preallocate consolidated column
pred_db[, consolidated := NA_real_]

# Create an index for joining later
pred_db[, row_id := .I]

# Split joined_data by year for chunk processing
joined_split <- split(joined_data, by = "year")

results_list <- vector("list", length(joined_split))

for (yr in names(joined_split)) {
  cat("Predicting for year", yr, "\n")
  
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    test_set <- joined_split[[yr]]
    
    # Predict in chunks to manage RAM
    chunk_size <- 50000L
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    results_list[[yr]] <- data.table(year = as.numeric(yr), id = test_set$id, pred = preds)
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Combine all predictions
predictions_dt <- rbindlist(results_list, use.names = TRUE, fill = TRUE)

# Join predictions back to pred_db
setkey(pred_db, year, id)
setkey(predictions_dt, year, id)
pred_db[predictions_dt, consolidated := i.pred]

# Write output efficiently
fwrite(pred_db[, !"row_id"], "RF_imputated_db.csv")
```

---

✅ **Why this is faster and memory-friendly:**  
- `data.table` provides efficient filtering and joining (O(1) subsets, minimal copies).  
- Chunked prediction prevents memory spikes.  
- Single join at the end avoids heavy in-loop assignments.  
- `fwrite()` is much faster than `write.csv()` for large datasets.  

This approach should make the process feasible on a standard personal computer without retraining the models.