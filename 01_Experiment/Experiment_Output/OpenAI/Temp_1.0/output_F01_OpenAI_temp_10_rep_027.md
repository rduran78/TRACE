 **Diagnosis**  
The bottlenecks are:  
1. **Large data frame filtering inside the loop** (`joined_data %>% filter(year == year)` repeatedly copies huge data).  
2. **`predict()` on very large `data.frame` objects** is memory- and time-intensive.  
3. Looping through all years with repeated filtering causes unnecessary overhead.  
4. Random Forest prediction is CPU-heavy; no parallelism used.  

**Optimization Strategy**  
- Avoid repeated filtering by **splitting data once by year** outside the loop.  
- Use `data.table` for fast subset operations and in-place updates.  
- Leverage **chunked predictions** to avoid memory blow-up.  
- Use `parallel::mclapply` or `future.apply` for parallelization (if multiple cores available).  
- Preserve trained models as requested.  

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Split joined_data by year once (as a list)
data_by_year <- split(joined_data, by = "year", keep.by = FALSE)

# Function to process each year
predict_for_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- data_by_year[[as.character(year)]]
    
    # Chunk prediction to reduce memory usage
    chunk_size <- 50000
    n <- nrow(test_set)
    preds <- numeric(n)
    
    for (i in seq(1, n, by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
    
    return(list(year = year, preds = preds))
  } else {
    warning(paste0("No model found for year ", year))
    return(NULL)
  }
}

# Parallel execution (adjust cores as needed)
years <- unique(pred_db$year)
results <- mclapply(years, predict_for_year, mc.cores = max(1, detectCores() - 1))

# Update pred_db with predictions
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

# Write to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this is faster and memory-safe:**  
- Data filtered once at the start → no repeated copy overhead.  
- Predictions done in chunks → avoids loading entire set into memory for `predict()`.  
- Parallel processing → utilizes multiple cores.  
- `data.table` for efficient joins and updates.  

This approach dramatically reduces runtime and memory footprint while preserving your trained Random Forest models.