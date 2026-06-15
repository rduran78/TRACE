 **Diagnosis:**  
The core bottlenecks are:
1. **`for` loop over years** performing prediction repeatedly on large subsets → inefficient in R.
2. Using `dplyr` filtering for each iteration → expensive for hundreds of thousands of rows.
3. Potentially large `pred_db` object being updated in a loop.
4. Predictions are done serially on a single core despite being easily parallelizable.
5. All objects stay in memory, risking memory pressure.

---

**Optimization Strategy:**
- Switch from iterative row updates to **vectorized or batch processing**.
- Use **`data.table`** for fast filtering and assignment (efficient in memory and time).
- Leverage **parallelization** using `parallel::mclapply` or `future.apply` since individual year predictions are independent.
- Avoid unnecessary copying by pre-splitting data by year once.
- Write the output incrementally if memory is tight.

---

### **Optimized R Code:**

```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

# Get unique years and split joined_data for faster access
year_list <- split(joined_data, by = "year")

# Detect number of cores for parallelization
n_cores <- max(1, detectCores() - 1)

# Parallel prediction
predictions <- mclapply(names(year_list), function(y) {
  dt <- year_list[[y]]
  if (y %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[y]]
    preds <- predict(rf_model, newdata = dt)
    return(data.table(id = dt$id, pred = preds))  # assuming an ID field exists
  } else {
    warning(paste("No model found for year", y))
    return(NULL)
  }
}, mc.cores = n_cores)

# Combine predictions
predictions_dt <- rbindlist(predictions, use.names = TRUE, fill = TRUE)

# Merge back with pred_db using fast join
setkey(pred_db, id)   # requires an 'id' key
setkey(predictions_dt, id)

pred_db[predictions_dt, consolidated := i.pred]

# Save result
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Gains:**  
- Single-pass join instead of repeated assignment.
- Data splitting and parallel prediction reduce time drastically.
- `data.table` avoids repeated copies and is memory-friendly.
- Model objects remain unchanged per your constraint.

To further improve memory usage, you could process year groups in **chunks** and write partial CSVs (then combine) if data is too large to hold entirely in memory.