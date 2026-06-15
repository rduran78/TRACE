 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), returning the **entire dataset** every iteration instead of a single year's subset. This is both a correctness bug and a massive performance problem — every year's prediction runs on the full dataset.

4. **`predict.randomForest` on huge data frames:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows at once with a large Random Forest model can spike memory because the full data frame (with all columns, including unused ones) is passed.

5. **Writing a potentially enormous CSV at the end:** `write.csv` is slow for large data; `data.table::fwrite` is significantly faster.

6. **`pred_db` is likely a data.frame:** Column assignment inside a loop (`pred_db$consolidated[pred_db$year == year] <- ...`) on a large data.frame triggers repeated copying. Using `data.table` with set-by-reference is far more efficient.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| Shapefile loaded but unused | Remove `st_read()` call |
| `filter(year == year)` masking bug | Use `.env$year` or rename the loop variable |
| Entire data.frame passed to `predict()` | Select only the required predictor columns |
| Slow row assignment in a loop | Convert to `data.table` and use `:=` by reference |
| Slow `write.csv` | Use `data.table::fwrite` |
| Optional: memory spikes | Process in chunks and call `gc()` between years |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)   # for predict.randomForest
library(data.table)     # fast data manipulation and fwrite
library(dplyr)          # for select (used once to get predictor names)

# ── Load the saved workspace (models + data) ─────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert working tables to data.table for speed ───────────────
setDT(pred_db)
setDT(joined_data)

# ── Initialise the output column ─────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Identify predictor columns from the first available model ────
# (avoids passing unnecessary columns to predict)
sample_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(importance(sample_model))

# ── Verify all predictors exist in joined_data ───────────────────
missing <- setdiff(predictor_vars, names(joined_data))
if (length(missing) > 0) {
  stop("Missing predictor columns in joined_data: ",
       paste(missing, collapse = ", "))
}

# ── Prediction loop (year by year) ───────────────────────────────
years <- sort(unique(pred_db$year))

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── Fixed filter: use `yr` (loop var) not `year` (column) ──────
  test_idx <- which(joined_data$year == yr)
  test_set <- joined_data[test_idx, ..predictor_vars]   # only needed columns

  preds <- predict(rf_model, newdata = test_set)

  # ── Assign by reference (no copy of pred_db) ───────────────────
  pred_db[year == yr, consolidated := preds]

  # ── Free memory between iterations ─────────────────────────────
  rm(test_set, preds)
  gc()
}

# ── Write output with fast fwrite ────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

- **Removed** all unused library imports and the unused `st_read()` call → **frees hundreds of MB of RAM**.
- **Fixed the critical filter bug** (`year == year` → `joined_data$year == yr`) → predictions are now correct and each iteration processes only one year's data instead of the entire dataset.
- **Subsetted to predictor columns only** (`..predictor_vars`) → `predict()` receives a lean matrix, reducing memory pressure.
- **Switched to `data.table`** with `:=` assignment → avoids repeated deep copies of `pred_db`.
- **Replaced `write.csv` with `fwrite`** → typically 5–10× faster for large files.
- **Added `gc()` per iteration** → encourages R to release memory between years.

These changes together should make the workflow feasible on a standard personal computer without retraining any models.