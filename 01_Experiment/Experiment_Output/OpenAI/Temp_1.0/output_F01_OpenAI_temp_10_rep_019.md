 **Diagnosis:**  
The bottleneck is the `for` loop applying predictions year by year on large datasets in-memory, which is inefficient for hundreds of thousands of rows and many predictors. Pulling the entire `joined_data` with each loop iteration and assigning predictions into `pred_db` adds memory and processing overhead.  

**Optimization Strategy:**  
- Avoid filtering large `joined_data` repeatedly; instead split or index once.  
- Use **data.table** for efficient subsetting and assignment.  
- Predict in **chunks** to reduce memory footprint.  
- Prevent unnecessary copying (e.g., avoid growing objects).  
- Keep models in memory (as required), no retraining.  

**Optimized R Code:**  
```r
library(data.table)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Prepare mapping: split joined_data by year for direct access
joined_split <- split(joined_data, by = "year")

# Predict in chunks to limit memory load
chunk_predict <- function(model, data_chunk) {
  predict(model, newdata = data_chunk)
}

for (yr in names(rf_models_per_year)) {
  if (!yr %in% names(joined_split)) next
  cat("Predicting for year", yr, "\n")
  
  rf_model <- rf_models_per_year[[yr]]
  dt_yr <- joined_split[[yr]]
  
  # Process in chunks (e.g., 50,000 rows per chunk)
  chunk_size <- 50000
  idx_seq <- seq(1, nrow(dt_yr), by = chunk_size)
  preds <- vector("numeric", nrow(dt_yr))
  
  for (i in seq_along(idx_seq)) {
    start <- idx_seq[i]
    end <- min(start + chunk_size - 1, nrow(dt_yr))
    preds[start:end] <- chunk_predict(rf_model, dt_yr[start:end])
  }
  
  pred_db[year == as.integer(yr), consolidated := preds]
}

fwrite(pred_db, "RF_imputated_db.csv")
```

**Why this works:**  
- Reduces repeated filtering (pre-split by year).  
- Uses `data.table` for efficient indexing and assignment.  
- Chunked prediction keeps memory manageable even on standard hardware.  
- Preserves trained models without any retraining.  

This approach is scalable, minimizes RAM pressure, and speeds up I/O.