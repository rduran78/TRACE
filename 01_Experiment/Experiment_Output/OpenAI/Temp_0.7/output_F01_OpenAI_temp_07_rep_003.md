 **Diagnosis:**  
The bottleneck is the loop performing `predict()` on very large yearly subsets. Loading the entire dataset into memory and predicting year by year without batching overwhelms memory and slows computation. Data frames and `dplyr` pipelines add overhead compared to more memory-efficient structures like `data.table`.  

---

**Optimization Strategy:**  
- Use `data.table` for fast filtering and assignment instead of `dplyr` pipes.  
- Predict in **batches (chunks)** to avoid loading the entire year’s data into memory at once.  
- Disable unnecessary features like row names in output and avoid repeated coercions.  
- Avoid re-allocating objects inside the loop; pre-allocate prediction vector.  

---

**Optimized R Code:**  
```r
library(data.table)

# Load shapefile (if needed for other tasks)
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')

# Load pre-trained models and data
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Pre-allocate prediction column
pred_db[, consolidated := NA_real_]

# Define batch size (adjust based on memory, e.g., 50k rows per chunk)
batch_size <- 50000

# Loop through years
for (year in unique(pred_db$year)) {
  cat("Predicting for year", year, "\n")
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Subset once for this year
    idx <- which(pred_db$year == year)
    test_set <- joined_data[year == !!year]
    
    # Process in chunks
    n <- nrow(test_set)
    for (i in seq(1, n, by = batch_size)) {
      j <- min(i + batch_size - 1, n)
      chunk <- test_set[i:j]
      preds <- predict(rf_model, newdata = chunk)
      pred_db$consolidated[idx[i:j]] <- preds
    }
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}

# Write output efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why This Helps:**  
- `data.table` dramatically reduces memory overhead and speeds subsetting/assignment.  
- Chunked prediction prevents memory exhaustion and enables processing even on modest machines.  
- Pre-allocation avoids repeated vector resizing.  

This approach preserves the original trained models and only optimizes the prediction workflow for scalability.