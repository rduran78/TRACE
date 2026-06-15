 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not work as intended. The loop variable `year` shadows the column name `year`, so the filter matches *all* rows every iteration instead of just the target year. This means the `predict()` call processes the entire dataset every iteration, massively inflating computation time.

4. **Predicting on the full dataset repeatedly:** Even if the filter were correct, calling `predict()` inside a loop year-by-year on a large `data.frame`/`tibble` is slower than necessary due to repeated subsetting and assignment overhead.

5. **Using `data.frame` for large assignment operations:** Assigning predictions back into `pred_db$consolidated[pred_db$year == year]` row-by-row-group on a large `data.frame` is slow. `data.table` would be faster.

6. **Writing a potentially huge CSV:** `write.csv` on hundreds of thousands (or millions) of rows is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and objects | Remove `st_read()` call and unneeded `library()` calls to free memory |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) |
| Whole-dataset predict per iteration | Fix the filter so only the relevant subset is predicted |
| Slow subsetting/assignment | Convert `pred_db` and `joined_data` to `data.table` |
| Slow CSV write | Use `fwrite()` instead of `write.csv()` |
| Optional: memory pressure from model list | Predict one year at a time and remove the model from the list after use with `gc()` |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and prediction data ──────────────────────────
# This .RData file is expected to contain:
#   - rf_models_per_year  (named list of trained RF models, keyed by year as character)
#   - joined_data         (data.frame/data.table with predictor columns + "year")
#   - pred_db             (data.frame/data.table that will receive predictions + "year")
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast operations ────────────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Predict year by year ─────────────────────────────────────────────────
years_to_predict <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Subset only the rows for this year (no variable-masking issue)
  test_set <- joined_data[year == as.numeric(yr)]

  # Predict using the pre-trained model (no retraining)
  preds <- predict(rf_models_per_year[[yr]], newdata = test_set)

  # Assign predictions back efficiently via data.table
  pred_db[year == as.numeric(yr), consolidated := preds]

  # Free the model from memory after use to reduce peak RAM
  rf_models_per_year[[yr]] <- NULL
  gc()
}

# Warn about any years with no model
missing_years <- setdiff(
  as.character(unique(pred_db$year)),
  years_to_predict
)
if (length(missing_years) > 0) {
  cat(paste0("Warning: No model found for year(s): ",
             paste(missing_years, collapse = ", "), "\n"))
}

# ── Write output efficiently ─────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Change | Benefit |
|---|---|
| Removed `st_read()` and unused libraries | Saves potentially **gigabytes** of RAM |
| Fixed `year == year` masking bug | Reduces per-iteration predict from **N total rows → N/year rows** (critical fix) |
| `data.table` for subsetting and assignment | ~5–10× faster than base `data.frame` on large data |
| `NULL`-ing models + `gc()` after each year | Frees each Random Forest from RAM immediately after use |
| `fwrite()` instead of `write.csv()` | ~5–20× faster file writing |

No models are retrained; only the prediction and I/O pipeline is optimized.