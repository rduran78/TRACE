 **Diagnosis:**  
The bottleneck is the loop over years combined with `predict()` on large datasets in memory. Each iteration processes hundreds of thousands of rows, which is memory-intensive. Additionally, `joined_data %>% filter(year == year)` repeatedly subsets a large data frame, adding overhead.  

**Optimization Strategy:**  
- Use `data.table` for efficient filtering and assignment instead of `dplyr`.
- Process predictions in chunks to reduce memory load.
- Avoid creating large intermediate objects; write predictions incrementally.
- Keep trained models intact and apply `predict()` directly on subsets.
- Use `fwrite()` for fast output instead of `write.csv()`.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Load data and models
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Initialize consolidated column
pred_db[, consolidated := NA_real_]

# Create output file and write header
output_file <- "RF_imputated_db.csv"
fwrite(pred_db[0], output_file)  # write empty structure for header

# Process predictions by year in chunks
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  # Efficient filtering
  test_set <- joined_data[year == yr]
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Chunk processing to avoid memory overload
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      end_i <- min(i + chunk_size - 1, n)
      preds[i:end_i] <- predict(rf_model, newdata = test_set[i:end_i])
    }
    
    # Assign predictions
    pred_db[year == yr, consolidated := preds]
    
    # Write incrementally to disk
    fwrite(pred_db[year == yr], output_file, append = TRUE)
    
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

cat("Prediction completed. Output saved to", output_file, "\n")
```

---

**Key Improvements:**  
- `data.table` for fast filtering and assignment.
- Chunked prediction reduces memory usage.
- Incremental writing prevents large in-memory objects.
- Preserves trained Random Forest models without retraining.  

This approach makes the process feasible on a standard machine while handling large datasets efficiently.