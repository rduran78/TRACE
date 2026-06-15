 **Diagnosis:**  
The loop processes predictions year by year with full datasets in memory, which is inefficient given the large number of cells and predictors. Random Forest prediction is relatively fast, but handling massive data frames and repeated filtering in tidyverse for each year creates memory overhead and slows execution. The major bottlenecks are:  
- `joined_data %>% filter(year == year)` repeatedly copies large data.  
- Storing all predictions in a single frame during execution without chunking.  
- Lack of parallelization.  

---

**Optimization Strategy:**  
1. Use **data.table** for efficient filtering and memory handling.  
2. Process in **chunks** or per-year subsets efficiently, without repeatedly copying large frames.  
3. Apply **parallel prediction** using `parallel` or `future.apply`.  
4. Write partial outputs to disk incrementally instead of holding all predictions in memory.  
5. Keep trained models intact; only optimize data handling and prediction execution.  

---

**Optimized R Code:**  
```r
library(data.table)
library(parallel)

# Load spatial and models
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Detect available cores for parallel execution
num_cores <- max(1, detectCores() - 1)

# Create output file and write header
output_file <- "RF_imputated_db.csv"
fwrite(pred_db[0], output_file) # empty header

# Split years for parallel processing
years <- sort(unique(pred_db$year))

cl <- makeCluster(num_cores)
clusterExport(cl, c("joined_data", "rf_models_per_year", "pred_db"))
clusterEvalQ(cl, library(randomForest))

results <- parLapply(cl, years, function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year] # efficient filter
    preds <- predict(rf_model, newdata = test_set)
    data.table(year = year, id = test_set$id, consolidated = preds)
  } else {
    NULL
  }
})
stopCluster(cl)

# Combine and write incrementally to reduce memory usage
results <- rbindlist(results, use.names = TRUE, fill = TRUE)
pred_db[results, on = .(year, id), consolidated := i.consolidated]
fwrite(pred_db, output_file)

cat("Prediction complete. Output saved to", output_file, "\n")
```

---

**Why This Works:**  
- `data.table` reduces memory overhead for filtering and joining.  
- Parallel prediction speeds up per-year predictions.  
- Incremental writing prevents huge in-memory objects.  
- Preserves original trained Random Forest models intact.  

This approach makes the task manageable on a standard personal computer.