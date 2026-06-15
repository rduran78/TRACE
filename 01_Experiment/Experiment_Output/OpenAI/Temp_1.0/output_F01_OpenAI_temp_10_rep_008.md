 **Diagnosis**  
The bottlenecks are:  
1. **Large data size and sequential processing per year** — Filtering and predicting year-by-year causes repeated data manipulation.  
2. **`tidyverse` pipelines and `predict()` in a loop** — These add overhead when working with hundreds of thousands of rows.  
3. **Single-thread prediction** — `randomForest::predict()` can be slow for large datasets without parallelization.  

**Optimization Strategy**  
- Use **data.table** for fast subsetting and assignment.  
- Combine all predictions into a single pass rather than separate loops.  
- Use **parallel processing** where possible (`parallel::mclapply` or `foreach` with `doParallel`).  
- Avoid re-filtering joined data multiple times. Pre-split once or work on keyed tables.  

**Optimized R Code**  
```r
library(data.table)
library(parallel)

# Load spatial and model data
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]  # Initialize column

years <- unique(pred_db$year)
ncores <- max(1, detectCores() - 1)  # Use available cores minus one

# Function to predict for a single year
predict_year <- function(yr) {
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    test_set <- joined_data[year == yr]
    preds <- predict(rf_model, newdata = test_set)
    list(year = yr, preds = preds)
  } else {
    NULL
  }
}

# Parallel execution
results <- mclapply(years, predict_year, mc.cores = ncores)

# Combine predictions back into pred_db
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

# Write output
fwrite(pred_db, "RF_imputated_db.csv")
```

**Key improvements**  
- **data.table** speeds up filtering and assignment dramatically.  
- `mclapply` enables parallel year-by-year prediction without retraining.  
- Eliminates unnecessary tidyverse overhead, preserving RAM.  

This approach ensures computational feasibility while retaining existing trained models.