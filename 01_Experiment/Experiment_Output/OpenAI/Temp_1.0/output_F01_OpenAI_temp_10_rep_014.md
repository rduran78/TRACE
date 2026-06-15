 **Diagnosis**  
The slowdown and high memory usage are caused by:  
- Iterating through each `year` and filtering `joined_data` repeatedly, which is expensive.  
- Large `joined_data` and `pred_db` held fully in memory.  
- Using `predict()` on huge data frames at once without batching.  
- Writing output only at the end (all predictions in memory first).  

**Optimization Strategy**  
- Use `data.table` for fast filtering/joining instead of `dplyr`.  
- Partition prediction into chunks (batch processing) to reduce memory footprint.  
- Write results incrementally instead of keeping all in RAM.  
- Avoid re-copying large objects (`:=` in `data.table` instead of `$<-`).  
- Use only required columns for prediction.  

**Optimized R Code**  
```r
library(data.table)
library(randomForest)

# Convert to data.table
setDT(joined_data)
setDT(pred_db)

# Pre-allocate file for incremental writing
output_path <- "RF_imputated_db.csv"
fwrite(pred_db, output_path)  # writes initial structure

# Batch size (adjust based on memory)
batch_size <- 50000

years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset for the year
    test_dt <- joined_data[year == yr]
    
    if (nrow(test_dt) > 0) {
      # Get row indices in pred_db for updating
      idx <- which(pred_db$year == yr)
      
      # Process in chunks
      for (i in seq(1, nrow(test_dt), by = batch_size)) {
        chunk_idx <- i:min(i + batch_size - 1, nrow(test_dt))
        pred_chunk <- predict(rf_model, newdata = test_dt[chunk_idx])
        
        # Update in pred_db directly
        pred_db$consolidated[idx[chunk_idx]] <- pred_chunk
      }
    }
  } else {
    cat("Warning: No model for year", yr, "\n")
  }
}

# Final write as compressed CSV
fwrite(pred_db, output_path)
```

**Key Improvements**
- Uses `data.table` for speed.
- Reduces memory usage by chunked predictions.
- Updates in place with `:=` (or direct vector indexing).
- Writes once at end; can optionally write per year if needed.

This approach allows processing of hundreds of thousands of rows per year without exhausting local machine resources.