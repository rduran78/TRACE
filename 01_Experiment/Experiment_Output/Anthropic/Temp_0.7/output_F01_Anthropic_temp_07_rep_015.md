 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning *all* rows every iteration. This means every prediction pass processes the entire dataset instead of a single year's subset—massively inflating computation time and memory use.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full data frame in memory:** `pred_db` and `joined_data` may be stored as standard `data.frame` or `tibble` objects. For hundreds of thousands of rows with many columns, subsetting and assignment via base R or dplyr can be slow and memory-hungry.

4. **Monolithic CSV write:** Writing a very large data frame to CSV at the end with `write.csv()` is slow; `data.table::fwrite()` is significantly faster.

5. **Loading unnecessary packages:** Many loaded libraries (e.g., `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`) are not used in the prediction step and consume memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Unnecessary libraries | Load only what is needed for prediction |
| Slow subsetting / assignment | Convert to `data.table` for fast keyed operations |
| Slow CSV export | Use `fwrite()` |
| Optional: memory pressure from predictors | Select only the predictor columns needed by the RF model before calling `predict()` |

---

## Optimized R Code

```r
# ── Load only the packages required for prediction ──────────────────────────
library(randomForest)
library(data.table)

# ── Load the saved workspace (contains rf_models_per_year, pred_db, joined_data)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast keyed operations ─────────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Initialise the target column ────────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Key both tables on year for fast subsetting ─────────────────────────────
setkey(pred_db,    year)
setkey(joined_data, year)

# ── Identify the predictor columns the RF models expect ─────────────────────
# (taken from any one of the stored models; all share the same feature set)
example_model <- rf_models_per_year[[1]]
predictor_cols <- rownames(example_model$importance)

# ── Predict year by year ────────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Subset only the rows for this year and only the needed predictor columns.
  # This avoids copying the entire data set and keeps memory use minimal.
  test_subset <- joined_data[.(yr), ..predictor_cols]

  # predict() returns a vector; assign directly via data.table reference semantics
  preds <- predict(rf_model, newdata = test_subset)
  pred_db[.(yr), consolidated := preds]

  # Free memory immediately

  rm(test_subset, preds, rf_model)
  gc()
}

# ── Write results using the fast fwrite() ───────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | What changed | Why it helps |
|---|---|---|
| 1 | Renamed loop variable to `yr`; subset with `joined_data[.(yr), ...]` | **Fixes the critical bug** where every year processed the full dataset |
| 2 | Removed `st_read()` and all unused library calls | Frees hundreds of MB of RAM |
| 3 | Converted to `data.table` with `setkey(year)` | Keyed subsetting and reference-assignment (`:=`) are orders of magnitude faster than dplyr filter + base `$<-` |
| 4 | Selected only `predictor_cols` for the test subset | Avoids copying irrelevant columns into `predict()` |
| 5 | Added `rm()` + `gc()` inside the loop | Releases each year's temporary objects before the next iteration |
| 6 | Replaced `write.csv()` with `fwrite()` | Typically 5–10× faster for large files |
| 7 | **No models were retrained** | All original `rf_models_per_year` objects are used as-is |