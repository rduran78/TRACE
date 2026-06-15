 **Diagnosis:**  
- The code processes predictions in a loop year by year, creating repeated `dplyr` filtering operations on a large dataset, which is slow and memory-intensive.  
- The `predict` call operates sequentially on large chunks without batching or parallelization.  
- `pred_db$consolidated` is filled using logical indexing per year, adding overhead.  
- `sf` objects and large data frames in memory may consume unnecessary RAM during repeated operations.  

---

**Optimization Strategy:**  
1. **Use `data.table` for fast subsetting and assignment instead of `dplyr`.**  
2. **Remove `sf` geometry if not needed for prediction** (geometry is heavy and not used in Random Forest predictions).  
3. **Leverage parallelization with `parallel` or `future.apply`** in the prediction loop.  
4. **Batch predictions if too large** to avoid memory overload.  
5. **Pre-convert `pred_db` and `joined_data` to `data.table` and set keys for fast join.**  

---

**Optimized R Code:**

```r
library(data.table)
library(parallel)

# Remove geometry if not needed
st_geometry(prep_data) <- NULL

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# Ensure keys for faster filtering
setkey(pred_db, year)
setkey(joined_data, year)

# Detect available cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Define prediction function for each year
predict_for_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[J(year)]  # Fast filter by key
    preds <- predict(rf_model, newdata = test_set)
    list(year = year, preds = preds)
  } else {
    message("Warning: No model for year ", year)
    NULL
  }
}

years <- unique(pred_db$year)

# Run in parallel
results <- mclapply(years, predict_for_year, mc.cores = n_cores)

# Combine predictions back into pred_db
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this is faster:**  
- Eliminates repetitive `dplyr` operations with faster `data.table` joins.  
- Removes spatial geometry to free memory.  
- Runs predictions in parallel, fully utilizing CPU cores.  
- Uses in-place assignment (`:=`) in `data.table` to minimize overhead.  

This approach should make prediction feasible on a normal machine while preserving the trained Random Forest models.