 **Diagnosis:**  
The bottleneck is the prediction loop over years with large datasets loaded entirely in memory. The use of `predict()` on huge `test_set` objects combined with `joined_data` filtering for each year is inefficient. The entire dataframe and shapefile are also kept in memory, leading to high RAM consumption.

---

**Optimization Strategy:**  
1. **Chunk processing**: Split large `test_set` into manageable chunks and predict in batches to reduce memory pressure.
2. **Efficient filtering**: Use `data.table` for fast subsetting instead of `dplyr`.
3. **Avoid copying large objects**: Precompute indices and avoid repeated `%>%` operations.
4. **Write incremental results**: Save predictions to disk progressively instead of holding all in RAM.

---

**Optimized R Code:**
```r
library(data.table)
library(randomForest)

# Load data and models
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Prepare output file
out_file <- "RF_imputated_db.csv"
fwrite(pred_db[0], out_file) # create empty file with headers

# Function for chunked prediction
predict_in_chunks <- function(model, data_chunk, chunk_size = 50000) {
  n <- nrow(data_chunk)
  preds <- vector("numeric", n)
  for (i in seq(1, n, by = chunk_size)) {
    end_i <- min(i + chunk_size - 1, n)
    preds[i:end_i] <- predict(model, newdata = data_chunk[i:end_i])
  }
  preds
}

# Loop over years
years <- unique(pred_db$year)
for (year in years) {
  cat("Predicting for year", year, "\n")
  
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    
    # Efficient filtering using data.table
    test_set <- joined_data[year == ..year]
    
    # Predict in chunks
    predictions <- predict_in_chunks(rf_model, test_set)
    
    # Assign and write incrementally
    pred_db[year == ..year, consolidated := predictions]
    fwrite(pred_db[year == ..year], out_file, append = TRUE)
  } else {
    cat("Warning: No model found for year", year, "\n")
  }
}
```

**Benefits:**  
- Uses `data.table` for fast filtering and assignment.
- Predicts in chunks (default 50k rows) to avoid memory overload.
- Writes results incrementally, reducing RAM usage.
- Preserves trained models without retraining.  

This approach will make the process computationally feasible on a normal machine.