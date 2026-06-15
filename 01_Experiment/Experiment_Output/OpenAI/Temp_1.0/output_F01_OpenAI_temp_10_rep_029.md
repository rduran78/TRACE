 **Diagnosis:**  
The bottleneck is the loop predicting year by year on potentially large `joined_data` subsets, with repeated filtering and in-memory operations on hundreds of thousands of rows and many columns. This causes inefficient memory use and slow speed. The `predict` function on `randomForest` models is fine, but applying it repeatedly without batching is costly.

---

**Optimization Strategy:**  
- Use **data.table** instead of `dplyr` for filtering and assignments (significantly faster and memory efficient).
- Avoid growing vectors repeatedly. Instead, preallocate prediction column.
- Use **parallel processing** for predictions by year (leveraging `parallel::mclapply` or `future.apply`).
- Write output incrementally instead of holding everything in memory.
- Only select relevant columns for prediction (drop unused variables).
- Keep models as is; do not retrain.

---

**Optimized R Code:**
```r
library(data.table)
library(parallel)

# Convert joined_data and pred_db to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Preallocate prediction column
pred_db[, consolidated := NA_real_]

# Get unique years
years <- unique(pred_db$year)

# Function for prediction per year
predict_year <- function(y) {
  message("Predicting for year ", y)

  if (as.character(y) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(y)]]

    # Filter once and select necessary predictors only
    test_set <- joined_data[year == y]

    # Drop non-predictor cols if necessary:
    test_set <- test_set[, names(rf_model$forest$xlevels), drop = FALSE]

    preds <- predict(rf_model, newdata = test_set)
    list(year = y, preds = preds)
  } else {
    warning("No model found for year ", y)
    NULL
  }
}

# Use parallel mclapply (adjust cores as needed)
results <- mclapply(years, predict_year, mc.cores = detectCores() - 1)

# Combine predictions back into pred_db
for (res in results) {
  if (!is.null(res)) {
    pred_db[year == res$year, consolidated := res$preds]
  }
}

# Write out efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this works better:**  
- `data.table` filtering and assignment is much faster than `dplyr`.
- Predictions run in parallel across years.
- Dropping irrelevant columns reduces memory load.
- Avoids repeated `tidyverse` overhead inside loops.
- Writes output efficiently with `fwrite`.

This approach scales well for hundreds of thousands of rows per year on a normal machine without retraining the models.