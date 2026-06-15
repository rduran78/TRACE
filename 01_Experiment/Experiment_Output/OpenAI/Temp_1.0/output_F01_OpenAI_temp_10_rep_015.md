 **Diagnosis:**  
The bottlenecks are:  
1. **Memory overhead** from loading and copying large `joined_data` each loop iteration.  
2. **Single-threaded looped prediction** for each year without batching or efficient data access.  
3. Using `predict()` on huge data frames without chunking overwhelms memory.  
4. `dplyr` operations and repeated filtering inside a loop are costly on large datasets.

---

**Optimization Strategy:**  
- Use `data.table` for efficient subsetting and assignment in place.  
- Perform predictions in **chunks** to reduce memory pressure.  
- Keep `rf_models_per_year` as is (no retraining).  
- Avoid repeated filtering; split once by year into a keyed `data.table`.  
- Enable **parallel prediction** if possible (`parallel::mclapply`).  

---

**Optimized R Code:**

```r
library(data.table)
library(randomForest)
library(parallel)

# Convert joined_data and pred_db to data.table
setDT(joined_data)
setDT(pred_db)
setkey(pred_db, year)

# Pre-split test sets by year for fast access
year_list <- split(joined_data, by = "year")

# Function for chunked prediction
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  for (i in seq(1, n, by = chunk_size)) {
    end_i <- min(i + chunk_size - 1, n)
    preds[i:end_i] <- predict(model, newdata = data[i:end_i])
  }
  preds
}

# Parallel loop over years
years <- names(rf_models_per_year)
mclapply(years, function(y) {
  cat("Predicting for year", y, "\n")
  if (y %in% names(year_list)) {
    test_set <- year_list[[y]]
    model <- rf_models_per_year[[y]]
    preds <- predict_in_chunks(model, test_set)
    pred_db[year == as.integer(y), consolidated := preds]
  } else {
    cat("Warning: Data not found for year", y, "\n")
  }
}, mc.cores = detectCores() - 1)

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**  
- `data.table` for memory-efficient filtering and assignment.  
- Chunked prediction (`chunk_size` adjustable to fit RAM).  
- Parallelization via `mclapply` to use multiple cores.  
- Avoid repeated subsetting by pre-splitting data.  

This reduces both memory footprint and runtime significantly while preserving trained models.