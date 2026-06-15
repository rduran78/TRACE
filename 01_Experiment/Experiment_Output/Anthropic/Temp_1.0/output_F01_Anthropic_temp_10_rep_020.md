 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Scoping Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), meaning the entire `joined_data` is loaded every iteration rather than just one year's subset. This is the single biggest problem — it causes every prediction call to process the *entire* dataset instead of a fraction.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Heavy Libraries Loaded Unnecessarily:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but not needed for prediction, consuming memory.

4. **Full Data Frame in Memory:** `joined_data` and `pred_db` may both be very large data frames held entirely in memory simultaneously.

5. **`predict.randomForest` on Massive Data:** Even with the bug fixed, predicting hundreds of thousands of rows at once with a large Random Forest can be memory-intensive.

6. **Writing a Massive CSV at the End:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Self-referencing filter bug | Use a temporary variable name: `yr` in the loop, filter with `.env$yr` or rename |
| Unused `st_read` call | Remove it |
| Unnecessary libraries | Load only what's needed |
| Memory pressure | Process year-by-year, and optionally write results incrementally |
| Slow CSV write | Use `data.table::fwrite` |
| Large single predict call | Optionally chunk predictions within a year |

---

## Optimized Working R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)
library(dplyr)

# ── Load the pre-trained models and associated data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: rf_models_per_year, joined_data, pred_db

# ── Remove any large objects from the .RData that are not needed ──
# (Uncomment and adjust if other large objects are present)
# rm(prep_data); gc()

# ── Convert to data.table for speed ──
setDT(pred_db)
setDT(joined_data)

# ── Initialize the output column ──
pred_db[, consolidated := NA_real_]

# ── Get the years to predict ──
years_to_predict <- unique(pred_db$year)

# ── Predict year by year ──
for (yr in years_to_predict) {
  cat(sprintf("Predicting for year %s ... ", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("WARNING: No model found. Skipping.\n")
    next
  }

  # ---- Subset only the rows for this year ----
  test_set <- joined_data[year == yr]

  if (nrow(test_set) == 0L) {
    cat("No rows in joined_data. Skipping.\n")
    next
  }

  # ---- Retrieve the pre-trained model (no retraining) ----
  rf_model <- rf_models_per_year[[yr_char]]

  # ---- Predict in chunks to limit peak memory ----
  n          <- nrow(test_set)
  chunk_size <- 50000L
  preds      <- numeric(n)

  for (start in seq(1L, n, by = chunk_size)) {
    end <- min(start + chunk_size - 1L, n)
    preds[start:end] <- predict(rf_model, newdata = test_set[start:end, ])
  }

  # ---- Write predictions back into pred_db ----
  pred_db[year == yr, consolidated := preds]

  cat(sprintf("%d rows predicted.\n", n))

  # ---- Free memory after each year ----
  rm(test_set, preds)
  gc()
}

# ── Write output with fast fwrite ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

* **Bug fix (critical):** The loop variable is now `yr`, so `joined_data[year == yr]` correctly subsets a single year instead of returning the entire table.
* **Removed unused `st_read` and unnecessary library calls**, freeing significant memory.
* **Switched to `data.table`** for faster subsetting and lower memory overhead.
* **Chunked predictions** (50,000 rows at a time) to cap peak memory during `predict()`.
* **Explicit `gc()` after each year** to release memory promptly.
* **Replaced `write.csv` with `fwrite`** for dramatically faster file output.
* **No models are retrained** — the existing `rf_models_per_year` objects are used as-is.