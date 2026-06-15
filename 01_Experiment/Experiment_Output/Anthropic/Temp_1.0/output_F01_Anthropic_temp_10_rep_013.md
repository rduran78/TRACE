 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (because the loop variable and column name are identical), returning **all rows** instead of the intended subset. This means every year's prediction runs on the entire dataset, massively inflating computation.

2. **Loading Unnecessary Spatial Data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes memory for no reason.

3. **Excessive Libraries:** Many loaded libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are not used in the prediction step, consuming memory overhead.

4. **Full Data Frame in Memory:** `joined_data` and `pred_db` may be very large data frames held entirely in memory. Subsetting with `dplyr::filter` on a non-indexed data frame is slow for hundreds of thousands of rows repeated across many years.

5. **Row-by-Row Assignment Pattern:** Assigning predictions back into `pred_db` via logical indexing (`pred_db$consolidated[pred_db$year == year]`) rescans the entire data frame each iteration.

6. **CSV Output for Large Data:** `write.csv` on a very large data frame is slow and produces large files; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename loop variable to `yr` |
| Unused spatial data loaded | Remove `st_read()` call |
| Unnecessary libraries | Load only what is needed |
| Slow subsetting & assignment | Convert to `data.table`, key by `year`, and use keyed joins |
| Slow CSV write | Use `data.table::fwrite()` |
| Potential memory pressure | Process and predict one year at a time; `gc()` between iterations |
| Large predict overhead | Keep only predictor columns needed by the model for each `predict()` call |

---

## Optimized Working R Code

```r
# ── Load only required libraries ──────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models ───────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert prediction database to data.table and key it ─────────
setDT(pred_db)
setDT(joined_data)

# Initialise the output column
pred_db[, consolidated := NA_real_]

# Key both tables by year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# ── Identify predictor columns once (from any stored model) ──────
sample_model   <- rf_models_per_year[[1]]
predictor_cols <- rownames(sample_model$importance)

# ── Predict year by year ─────────────────────────────────────────
years <- sort(unique(pred_db$year))

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # Fast keyed subset – only the columns the model needs
  test_set <- joined_data[.(yr), ..predictor_cols]

  # Predict
  preds <- predict(rf_models_per_year[[yr_char]], newdata = test_set)

  # Fast keyed assignment
  pred_db[.(yr), consolidated := preds]

  # Free memory from this iteration
  rm(test_set, preds)
  gc()
}

# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

1. **Bug fix:** Renamed the loop variable from `year` to `yr` so the filter/subset correctly selects only the rows for that year.
2. **Removed unused code:** Eliminated the `st_read` call and all libraries not required for prediction.
3. **Switched to `data.table`:** Keyed subsetting (`[.(yr)]`) and column restriction (`..predictor_cols`) are much faster and more memory-efficient than `dplyr::filter` on large frames.
4. **Column pruning:** Only the predictor columns required by the Random Forest model are passed to `predict()`, reducing memory copying.
5. **Explicit garbage collection** (`gc()`) after each year frees temporary allocations.
6. **`fwrite` instead of `write.csv`:** Typically 5–10× faster for large outputs.

These changes should make the workflow feasible on a standard personal computer without retraining any models.