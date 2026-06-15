 **Diagnosis:**  
The bottleneck comes from iterating through each year and predicting on large subsets, which causes repeated filtering and large in-memory operations. Hundreds of thousands of rows combined with many predictors and Random Forest prediction makes this memory-heavy and slow on a personal computer.

**Optimization Strategy:**  
1. Avoid repeated filtering with `dplyr` inside the loop; use `data.table` for fast subsetting.  
2. Process predictions in chunks (batch prediction) instead of loading entire yearly subsets into memory at once.  
3. Use `data.table::fread`/`fwrite` for efficient I/O instead of `write.csv`.  
4. Keep trained models unchanged but utilize parallel prediction where possible (`predict` on RF supports `ncores` in some implementations or use `parallel::mclapply`).  
5. Convert `pred_db` to `data.table` for efficient assignment.

---

**Optimized R Code:**
```r
library(data.table)
library(randomForest)
library(parallel)

# Load data and models
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table
setDT(joined_data)
setDT(pred_db)

# Ensure pred_db has 'consolidated' column
pred_db[, consolidated := NA_real_]

years <- unique(pred_db$year)

# Define chunk size (e.g., 50,000 rows)
chunk_size <- 50000

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset once using data.table
    test_set <- joined_data[year == yr]
    n <- nrow(test_set)
    
    # Process in chunks
    idx <- seq(1, n, by = chunk_size)
    results <- vector("numeric", n)
    
    for (i in idx) {
      end_i <- min(i + chunk_size - 1, n)
      chunk <- test_set[i:end_i]
      results[i:end_i] <- predict(rf_model, newdata = chunk)
    }
    
    # Assign back efficiently
    pred_db[year == yr, consolidated := results]
    
  } else {
    cat(sprintf("Warning: No model found for year %s\n", yr))
  }
}

# Efficient write
fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- `data.table` speeds up filtering and assignment.  
- Chunking reduces peak memory usage and avoids full-load bottlenecks.  
- Parallelism can be added by replacing the inner chunk loop with `mclapply` if desired.  

This approach preserves trained models, minimizes memory load, and is feasible for large datasets on a personal computer.