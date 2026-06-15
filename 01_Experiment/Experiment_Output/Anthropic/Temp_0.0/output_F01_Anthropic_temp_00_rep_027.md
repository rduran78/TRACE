 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly because the column name `year` and the loop variable `year` share the same name. This causes the filter to return **all rows** every iteration instead of a single year's subset, massively inflating computation per iteration.

4. **Whole-dataframe assignment in a loop:** Writing predictions back into `pred_db$consolidated` row-by-row (year-by-year) on a very large data.frame is inefficient. Using `data.table` would be faster.

5. **`predict.randomForest` on huge data:** With hundreds of thousands of rows and many trees, `predict()` can spike memory. This is unavoidable per year, but fixing the filter bug (point 3) ensures only one year's worth of data is passed at a time.

6. **Writing a massive CSV:** `write.csv` on a very large data.frame is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the unused `st_read()` call | Frees significant RAM |
| Fix the variable masking bug in `filter()` | Prevents predicting on the entire dataset each iteration |
| Convert `pred_db` and `joined_data` to `data.table` | Faster subsetting and assignment |
| Use `data.table::fwrite` instead of `write.csv` | Much faster I/O |
| Add `gc()` after each year | Reclaims memory between iterations |
| Optionally, remove `joined_data` subsets after use | Keeps peak memory lower |

The trained Random Forest models (`rf_models_per_year`) are **preserved and never retrained**.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)       # kept only if joined_data / pred_db are tibbles

# ── Load the pre-trained models and associated data ──────────────────
# (Do NOT load the unused spatial shapefile)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed ─────────────────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Prediction loop (models are NOT retrained) ──────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correct filter: use the loop variable 'yr', not 'year'
  test_set <- joined_data[year == yr]

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]
    preds    <- predict(rf_model, newdata = test_set)

    # Fast in-place update by reference
    pred_db[year == yr, consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each iteration
  rm(test_set, preds)
  gc()
}

# ── Write output with fast I/O ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Effect |
|---|--------|--------|
| 1 | Removed 10 unused `library()` calls | Lower memory footprint at startup |
| 2 | Removed unused `st_read()` of shapefile | Saves potentially gigabytes of RAM |
| 3 | Renamed loop variable to `yr` to fix `filter(year == year)` masking bug | Each iteration now processes **one year** instead of the entire dataset |
| 4 | Switched to `data.table` with `:=` assignment by reference | Faster subsetting and in-place update; avoids copying |
| 5 | Replaced `write.csv` with `fwrite` | Dramatically faster file writing |
| 6 | Added `rm()` + `gc()` inside loop | Reclaims memory between years |
| 7 | Pre-trained RF models are loaded and used as-is | No retraining occurs |

These changes should make the workflow feasible on a standard personal computer without altering the modeling results.