 **Diagnosis:**  
The bottleneck is iterating over years and predicting on large datasets (`joined_data`) without optimization. Each iteration performs a full subset and prediction, leading to high memory usage and slow performance due to repeated data frame operations in `dplyr`.

---

**Optimization Strategy:**  
1. Use `data.table` for efficient subsetting and joining instead of `dplyr`.  
2. Avoid repeated writes—construct prediction in a vectorized fashion.  
3. Process in chunks if data is too large to fit in RAM.  
4. Preconvert models and data to appropriate structures to minimize conversion overhead.  
5. Leverage multi-core prediction using `parallel::mclapply` or `future.apply` if OS supports.  

---

**Optimized R Code:**  
```r
library(data.table)
library(randomForest)
library(parallel)

# Load data efficiently
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

setDT(pred_db)
setDT(joined_data)

# Ensure columns match
pred_db[, consolidated := NA_real_]

years <- unique(pred_db$year)

# Parallel processing across years (adjust cores)
n_cores <- max(1, detectCores() - 1)

pred_results <- mclapply(years, function(y) {
  model_name <- as.character(y)
  if (model_name %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[model_name]]
    test_set <- joined_data[year == y]
    
    # Chunk prediction if very large
    chunk_size <- 100000  # adjust depending on memory
    n <- nrow(test_set)
    preds <- numeric(n)
    
    if (n > chunk_size) {
      idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
      for (i in seq_along(idx)) {
        preds[idx[[i]]] <- predict(rf_model, newdata = test_set[idx[[i]], ])
      }
    } else {
      preds <- predict(rf_model, newdata = test_set)
    }
    
    data.table(id = test_set$id, consolidated = preds) # assumes 'id' is unique key
  } else {
    NULL
  }
}, mc.cores = n_cores)

# Combine results and join back
pred_dt <- rbindlist(pred_results, use.names = TRUE)
setkey(pred_dt, id)
setkey(pred_db, id)

pred_db[pred_dt, consolidated := i.consolidated]

fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Gains:**  
- `data.table` for fast subsetting & joins.  
- Chunked predictions prevent overload.  
- Parallel prediction speeds up across years.  
- Single write operation minimizes I/O overhead.