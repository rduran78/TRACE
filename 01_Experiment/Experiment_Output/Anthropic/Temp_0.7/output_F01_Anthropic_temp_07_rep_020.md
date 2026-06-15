 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Unnecessary heavy library loading:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Unnecessary spatial data read:** `prep_data` is read from a shapefile but never used. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` becomes the **entire dataset** every iteration — massively inflating computation and producing incorrect results.

4. **Full data frame copy on every assignment:** Assigning into `pred_db$consolidated[pred_db$year == year]` inside a loop over years triggers repeated full-column scans and copy-on-modify behavior in base R / tibble data frames.

5. **`predict.randomForest` on massive data:** With hundreds of thousands of rows and many trees, `predict()` is memory-intensive. This is unavoidable per year but is worsened by the bug above (predicting on the full dataset each time).

6. **Using `tidyverse` tibbles for large data:** Tibbles are slower than `data.table` for row-subsetting and assignment operations at scale.

---

## Optimization Strategy

| # | Action | Impact |
|---|--------|--------|
| 1 | Remove unused libraries and the unused `st_read` call | Frees hundreds of MB of RAM |
| 2 | Fix the variable masking bug (`year == year`) | Reduces per-iteration prediction set from N to N/T (critical) |
| 3 | Convert working data to `data.table` | Faster subsetting and in-place assignment (no copy-on-modify) |
| 4 | Select only the predictor columns needed by the RF model before calling `predict()` | Reduces memory footprint of each `test_set` |
| 5 | Explicitly call `gc()` after each year to reclaim memory | Helps on RAM-constrained machines |
| 6 | Write output with `data.table::fwrite` | Much faster than `write.csv` for large files |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and prediction database ───────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Objects expected in workspace after load:
#   rf_models_per_year  – named list of randomForest models (one per year)
#   pred_db             – data.frame / tibble with a "year" column
#   joined_data         – data.frame / tibble with predictors + "year" column

# ── Convert to data.table for speed and in-place assignment ───────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Identify predictor columns from the first available model ─────
sample_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(sample_model$importance)

# ── Predict year by year ──────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  model_key <- as.character(yr)

  if (model_key %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[model_key]]

    # Subset rows for this year only (bug-fixed; uses 'yr', not 'year')
    # Select only the columns the model actually needs
    test_set <- joined_data[year == yr, ..predictor_vars]

    preds <- predict(rf_model, newdata = test_set)

    # In-place update by reference (no copy of the whole column)
    pred_db[year == yr, consolidated := preds]

    # Free temporary objects and reclaim memory
    rm(test_set, preds, rf_model)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

- **Removed** all unused `library()` calls and the unused `st_read()`, freeing substantial memory.
- **Fixed the critical bug:** renamed the loop variable to `yr` so `joined_data[year == yr, ...]` correctly filters to a single year instead of returning the entire dataset.
- **Switched to `data.table`:** enables column assignment by reference (`:=`) avoiding R's copy-on-modify, and provides fast row filtering via binary-optimized `[`.
- **Selected only predictor columns** (`..predictor_vars`) so each `test_set` is as small as possible.
- **Added `rm()` + `gc()`** per iteration to keep peak memory low.
- **Replaced `write.csv` with `fwrite`** for significantly faster output writing.

These changes preserve the trained Random Forest models exactly as-is and require no retraining.