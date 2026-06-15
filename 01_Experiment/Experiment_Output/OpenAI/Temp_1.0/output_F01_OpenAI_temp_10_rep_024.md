 **Diagnosis:**  
The bottleneck comes from iterating over years and applying `predict()` on large data sets loaded entirely into memory. The current design explicitly filters data for each year in-memory, which is heavy for hundreds of thousands of rows. Using `dplyr` and base loops makes repeated data subsetting expensive. The process needs chunked/batched predictions and memory-efficient data handling.

---

**Optimization Strategy:**  
1. **Switch to `data.table` for filtering and assignment** — very fast for large data.  
2. **Predict in chunks** to avoid loading entire year data at once into memory.  
3. **Write results incrementally** instead of storing a huge object until the end.  
4. Use `gc()` cleanup after each loop iteration if memory is tight.  

---

**Optimized R Code:**

```r
library(data.table)
library(randomForest)

# Read shapefile as data.table for efficiency
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')

# Load RF models
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table
setDT(joined_data)
setDT(pred_db)

# Add consolidated column if missing
if (!"consolidated" %in% names(pred_db)) pred_db[, consolidated := NA_real_]

# Prepare incremental CSV write
output_file <- "RF_imputated_db.csv"
fwrite(pred_db[0], output_file) # write header only

# Loop through years
years <- unique(pred_db$year)
for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  # Subset using data.table for speed
  test_set <- joined_data[year == yr]
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Predict in chunks of 50,000 rows
    chunk_size <- 50000
    preds <- numeric(nrow(test_set))
    for (start in seq(1, nrow(test_set), by = chunk_size)) {
      end <- min(start + chunk_size - 1, nrow(test_set))
      preds[start:end] <- predict(rf_model, newdata = test_set[start:end])
    }
    
    # Assign predictions to pred_db using direct matching
    pred_db[year == yr, consolidated := preds]
    
    # Write incrementally
    fwrite(pred_db[year == yr], output_file, append = TRUE)
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
  
  rm(test_set, preds)
  gc() # clean up
}

cat("Prediction complete. Output saved to", output_file, "\n")
```

**Why this works:**  
- `data.table` operations significantly reduce filtering overhead.  
- Chunked predictions prevent memory overflow.  
- Incremental `fwrite()` avoids holding the full result in RAM.  

This approach will make predictions on very large datasets feasible on a personal machine while preserving trained models.