 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. Spatial data (`.shp`) with hundreds of thousands of features is very memory-intensive.

3. **Variable shadowing bug:** Inside `filter(year == year)`, the column `year` and the loop variable `year` share the same name. This causes `dplyr::filter` to evaluate `year == year` as always `TRUE`, meaning `test_set` contains **all rows** every iteration — not just the target year. This multiplies memory use and prediction time by the number of unique years.

4. **Predicting on the full dataset every iteration:** Because of the bug above, the Random Forest `predict()` call processes the entire dataset in every loop pass instead of a single year's subset.

5. **Writing a potentially huge CSV at the end:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is significantly faster.

6. **No garbage collection or memory management:** Large intermediate objects persist in memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` spatial object | Remove the `st_read()` call entirely |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` or use `data.table` subsetting |
| Full-data prediction each iteration | Correctly subset so only one year's rows are predicted at a time |
| Slow CSV write | Use `data.table::fwrite()` |
| Memory pressure | Use `data.table` for subsetting; call `gc()` after each iteration; optionally convert `joined_data` and `pred_db` to `data.table` |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)

# ── Load the saved workspace containing rf_models_per_year, joined_data, pred_db ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast subsetting and assignment ──
if (!is.data.table(pred_db))      setDT(pred_db)
if (!is.data.table(joined_data))  setDT(joined_data)

# ── Initialize the output column ──
pred_db[, consolidated := NA_real_]

# ── Get the years for which a trained model exists ──
available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

# ── Predict year by year ──
for (yr in available_years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correct subset: compare column to the *loop variable* (no shadowing)
  year_num <- as.numeric(yr)
  test_set <- joined_data[year == year_num]

  # Predict using the pre-trained model for this year
  rf_model <- rf_models_per_year[[yr]]
  preds    <- predict(rf_model, newdata = test_set)

  # Assign predictions back (fast data.table update by reference)
  pred_db[year == year_num, consolidated := preds]

  # Free memory
  rm(test_set, preds, rf_model)
  gc()
}

# ── Warn about years with no model ──
missing_years <- setdiff(as.character(unique(pred_db$year)),
                         names(rf_models_per_year))
if (length(missing_years) > 0) {
  cat("Warning: No model found for year(s):", paste(missing_years, collapse = ", "), "\n")
}

# ── Write output using fast fwrite ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Change | Estimated Impact |
|---|---|
| Fix variable-shadowing bug | **Largest gain** — reduces rows processed per iteration from N (all years) to N/T (one year). For 20 years this is a ~20× speedup and ~20× memory reduction per iteration. |
| Remove `st_read` of unused shapefile | Saves potentially gigabytes of RAM |
| Remove unused libraries | Saves hundreds of MB of RAM |
| `data.table` subsetting & update-by-reference | Avoids full-copy semantics of `dplyr`; faster and lower memory |
| `fwrite` instead of `write.csv` | ~5–10× faster file writing |
| `gc()` each iteration | Returns freed memory to the OS promptly |

These changes preserve all trained Random Forest models exactly as-is (no retraining) and should make the workflow feasible on a standard personal computer.