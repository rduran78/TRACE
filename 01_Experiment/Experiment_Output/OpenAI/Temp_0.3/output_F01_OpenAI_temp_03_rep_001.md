 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large datasets using `predict()` with `randomForest`. Each iteration processes a large `data.frame` in memory, causing high RAM usage and slow execution. Additionally, `dplyr` operations inside the loop and repeated subsetting are inefficient for large datasets.

---

**Optimization Strategy:**  
1. **Avoid repeated filtering with `dplyr`**: Use `data.table` for efficient subsetting and assignment.
2. **Process in chunks**: Predict in manageable batches to reduce memory pressure.
3. **Pre-split data by year**: Avoid filtering inside the loop.
4. **Use `data.table::fread/fwrite`** for fast I/O.
5. **Parallelize predictions** if possible (optional).

---

**Optimized R Code:**
```r
library(data.table)
library(randomForest)

# Load data
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Pre-split joined_data by year for faster access
year_list <- split(joined_data, by = "year")

# Initialize consolidated column
pred_db[, consolidated := NA_real_]

# Define a function for chunked prediction
predict_in_chunks <- function(model, newdata, chunk_size = 50000) {
  n <- nrow(newdata)
  preds <- numeric(n)
  idx <- seq(1, n, by = chunk_size)
  for (i in idx) {
    end <- min(i + chunk_size - 1, n)
    preds[i:end] <- predict(model, newdata = newdata[i:end])
  }
  preds
}

# Loop through years efficiently
for (yr in names(year_list)) {
  cat("Predicting for year", yr, "\n")
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    test_set <- year_list[[yr]]
    preds <- predict_in_chunks(rf_model, test_set)
    pred_db[year == as.integer(yr), consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**  
- `data.table` for efficient filtering and assignment.
- Chunked prediction to avoid memory overload.
- Pre-splitting by year to eliminate repeated filtering.
- `fwrite` for fast output.

This approach will significantly reduce memory usage and speed up execution on a normal machine without retraining models.