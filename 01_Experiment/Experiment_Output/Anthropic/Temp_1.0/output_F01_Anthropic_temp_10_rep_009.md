 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a potentially massive `.RData` file entirely into memory:** The file `model_5_all_countries.RData` likely contains training data (`joined_data`), all year-specific Random Forest models (`rf_models_per_year`), and the prediction database (`pred_db`). All of these reside in memory simultaneously.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which means it evaluates to `TRUE` for every row (self-comparison) instead of filtering to the current loop iteration's year. This returns the entire dataset every iteration, causing the Random Forest `predict()` call to process all rows instead of one year's subset, drastically increasing computation time and producing incorrect results.

4. **Whole-dataframe column assignment in a loop:** Writing into `pred_db$consolidated[pred_db$year == year]` inside a loop over years is slow for large data frames due to repeated memory copying.

5. **`predict.randomForest` on hundreds of thousands of rows** is CPU-intensive and single-threaded by default; chunking or parallelism is not used.

6. **`st_read` for `prep_data`:** A large shapefile is loaded but never used in the prediction code, wasting memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and objects | Remove unused `library()` calls; remove `prep_data` load; after extracting needed objects from `.RData`, remove unneeded ones with `rm()` and `gc()` |
| Variable masking bug | Rename the loop variable (e.g., `yr`) so `filter(year == yr)` works correctly |
| Whole-frame writes in a loop | Collect predictions into a pre-allocated list, then bind once |
| Memory pressure from large `joined_data` | Convert `joined_data` to `data.table`; subset by year using keyed fast lookups; drop columns not in the model |
| Single-threaded `predict()` | Process years in parallel via `future.apply` or chunk large years |
| CSV output of huge frame | Use `data.table::fwrite()` for much faster writing |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────────
library(randomForest)
library(data.table)
library(dplyr)            # only for select/filter if needed

# ── 1. Load saved models and data ────────────────────────────────────────────
# This loads rf_models_per_year, joined_data, pred_db (and possibly others)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Identify predictor columns the models expect ─────────────────────────
# Use the first available model to get the list of required predictors
sample_model  <- rf_models_per_year[[1]]
required_vars <- rownames(importance(sample_model))

# ── 3. Trim joined_data to only required columns + year ─────────────────────
keep_cols <- intersect(c("year", required_vars), names(joined_data))
joined_dt <- as.data.table(joined_data)[, ..keep_cols]
setkey(joined_dt, year)

# Free the original large copy
rm(joined_data)
gc()

# ── 4. Convert pred_db to data.table for fast indexed writes ────────────────
pred_dt <- as.data.table(pred_db)
pred_dt[, consolidated := NA_real_]
setkey(pred_dt, year)

rm(pred_db)
gc()

# ── 5. Predict year-by-year (correctly) ─────────────────────────────────────
years_to_predict <- sort(unique(pred_dt$year))
model_years      <- names(rf_models_per_year)

for (yr in years_to_predict) {
  yr_char <- as.character(yr)
  cat(paste0("Predicting for year ", yr, " ... "))

  if (yr_char %in% model_years) {
    # Fast keyed subset – only rows for this year
    test_set <- joined_dt[.(yr)]

    rf_model <- rf_models_per_year[[yr_char]]

    preds <- predict(rf_model, newdata = test_set)

    # Fast update by reference (no copy)
    pred_dt[.(yr), consolidated := preds]

    cat(paste0(nrow(test_set), " cells done.\n"))

    # Free per-iteration objects
    rm(test_set, preds)
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── 6. Write output efficiently ─────────────────────────────────────────────
fwrite(pred_dt, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Bug fix:** Renamed the loop iterator to `yr` so `joined_dt[.(yr)]` correctly filters to one year instead of returning the entire dataset.
- **Memory:** Removed unused library loads and the unused `prep_data` shapefile. Dropped unneeded columns from `joined_data`. Used `rm()` and `gc()` aggressively.
- **Speed:** Converted data frames to `data.table` with keys for O(log n) subsetting. Used `:=` assignment by reference (zero-copy update) instead of base R indexed assignment. Replaced `write.csv` with `fwrite`.
- **Models preserved:** No models are retrained; `rf_models_per_year` is loaded and used as-is.

These changes alone should reduce memory usage substantially and cut prediction time significantly, making the workflow feasible on a standard personal computer.