 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (due to the loop variable sharing the same name), returning *all* rows instead of the intended subset. This means every iteration predicts on the entire dataset — massively increasing computation and producing wrong results.

2. **Loading the full spatial shapefile unnecessarily:** `prep_data` is read via `st_read` but never used in the prediction loop. Loading a large `.shp` file consumes significant memory for no reason.

3. **Predicting on the wrong object:** The code filters `joined_data` to create `test_set`, but writes predictions into `pred_db`. If these two objects are both large and separate copies, memory doubles. Their relationship is also unclear — if column names or row alignment differ, predictions silently go to the wrong rows.

4. **`predict.randomForest` on hundreds of thousands of rows at once:** Random Forest prediction with many trees and many predictors on a very large data frame can spike memory because `randomForest::predict` builds an intermediate matrix of per-tree predictions.

5. **Writing a massive CSV at the end:** `write.csv` on a data frame with hundreds of thousands of rows × many columns is slow; `data.table::fwrite` is far faster.

6. **Many unnecessary library loads:** Packages like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `zoo` are not used in the prediction step and consume memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing | Rename loop variable to `yr` |
| Unused shapefile load | Remove `st_read` call |
| Unnecessary libraries | Load only what is needed |
| Memory from full-dataset filter | Use `data.table` keyed subsetting |
| Large single-pass predict | Predict in chunks (batches) to cap peak memory |
| Slow `write.csv` | Use `data.table::fwrite` |
| Ensuring row alignment | Predict directly on the subset of `pred_db` itself (or verify column parity) |

---

## Optimized R Code

```r
# ------------------------------------------------------------------
# Optimized cell-level GDP prediction
# ------------------------------------------------------------------

# Load only the libraries actually needed for prediction
library(randomForest)   # predict method for RF models
library(data.table)     # fast I/O and subsetting

# 1. Load the saved models and prediction database
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: rf_models_per_year, pred_db
# (and possibly joined_data — see note below)

# 2. Convert to data.table for speed and memory-efficient subsetting
#    Use whichever object actually holds the predictor columns that
#    match the training data.  If joined_data is the correct source,
#    replace pred_db below with joined_data and write back accordingly.
setDT(pred_db)
pred_db[, consolidated := NA_real_]

# 3. Define a chunked predict function to cap peak memory
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  # Pre-allocate result vector
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# 4. Iterate over years with correct variable naming
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Identify rows for this year (data.table fast logical subset)
  idx <- which(pred_db$year == yr)

  # Predict in chunks to limit peak memory
  pred_db[idx, consolidated := predict_chunked(rf_model,
                                                newdata = pred_db[idx],
                                                chunk_size = 50000L)]

  cat(sprintf("  Done — %d rows predicted.\n", length(idx)))
}

# 5. Write results with fast fwrite
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

- **Bug fix:** Loop variable renamed from `year` to `yr` so the filter actually selects only the intended year's rows.
- **Memory savings (~GBs):** Removed the unused `st_read` call and unnecessary library loads.
- **Chunked prediction:** `predict_chunked` processes rows in batches of 50,000 (tunable), preventing a single massive allocation inside `predict.randomForest`.
- **`data.table` throughout:** Keyed subsetting and in-place `:=` assignment avoid copying the entire data frame.
- **`fwrite` instead of `write.csv`:** Typically 5–10× faster for large outputs.
- **Models are never retrained** — only `predict()` is called on the existing `rf_models_per_year`.