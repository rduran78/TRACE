 **Diagnosis:**  
The main bottleneck is the loop-based prediction over large datasets (`hundreds of thousands of cells per year`) combined with `dplyr` filtering and `predict()` calls for each year. This approach repeatedly subsets large data frames and performs predictions sequentially, which is slow and memory-intensive.  

**Optimization Strategy:**  
1. **Avoid repeated filtering:** Use `split()` once to create year-based subsets instead of filtering in a loop.  
2. **Use `data.table` for efficiency:** It handles large datasets faster than `dplyr`.  
3. **Parallelize predictions:** Use `parallel::mclapply` or `future.apply` to predict in parallel.  
4. **Preallocate results:** Avoid growing objects inside loops.  
5. **Keep models unchanged:** Use existing `rf_models_per_year`.  

---

### **Optimized R Code**
```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Split joined_data by year once
data_split <- split(joined_data, joined_data$year)

# Function for prediction per year
predict_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    preds <- predict(rf_model, newdata = data_split[[as.character(year)]])
    return(data.table(year = year, preds = preds))
  } else {
    warning(paste("No model found for year", year))
    return(NULL)
  }
}

# Parallel prediction
years <- names(data_split)
num_cores <- max(1, detectCores() - 1)
results_list <- mclapply(years, predict_year, mc.cores = num_cores)

# Combine predictions
results <- rbindlist(results_list, use.names = TRUE, fill = TRUE)

# Merge predictions back to pred_db
pred_db[results, consolidated := preds, on = .(year)]

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works:**  
- `split()` avoids repeated filtering.  
- `data.table` speeds up joins and memory handling.  
- `mclapply` uses multiple cores for parallel predictions.  
- No retraining occurs; models are reused as requested.  

This approach will significantly reduce runtime and memory overhead on a standard machine.