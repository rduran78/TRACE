 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` always equals the entire `joined_data` rather than a single year's subset. This means every iteration predicts on the full dataset — massively increasing compute time and producing incorrect results.

3. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` performs repeated logical indexing on a potentially huge data frame inside a loop, which is slow in base R.

4. **Using `data.frame` for a large dataset:** Standard data frames are slower than `data.table` for subsetting and assignment operations on hundreds of thousands of rows.

5. **Loading many unused libraries:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, and `zoo` are loaded but not used in this script, consuming memory.

6. **Writing output as CSV for a very large dataset:** For hundreds of thousands of rows × many columns, `write.csv()` is slow; `fwrite()` from `data.table` is much faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read()` call | Remove it entirely |
| Unused library loading | Load only `randomForest`, `dplyr` (or `data.table`), and `data.table` |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) |
| Slow row-level assignment in a loop | Convert to `data.table` and use keyed assignment by reference |
| Slow CSV write | Use `data.table::fwrite()` |
| Optional: memory pressure from large `.RData` | After extracting what's needed, remove unused objects with `rm()` and call `gc()` |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and prediction data ───────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: rf_models_per_year, joined_data, pred_db

# ── Convert to data.table for fast operations ─────────────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Remove any large objects from the .RData that are not needed ──────────
# (Adjust names to whatever else was saved in the file)
# rm(other_unneeded_object)
gc()

# ── Initialise the output column ──────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Predict year by year ──────────────────────────────────────────────────
available_years <- names(rf_models_per_year)

for (yr in unique(pred_db$year)) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% available_years) {
    # Correct subsetting — no variable-shadowing bug
    test_set <- joined_data[year == yr]

    rf_model <- rf_models_per_year[[yr_char]]
    preds    <- predict(rf_model, newdata = test_set)

    # Fast update by reference
    pred_db[year == yr, consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory each iteration
  rm(test_set, preds)
  gc()
}

# ── Write results with fast I/O ──────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Change | Benefit |
|---|---|
| Removed `st_read()` and unused libraries | Saves potentially gigabytes of RAM |
| Fixed `filter(year == year)` shadowing bug | Each iteration now processes only one year's rows instead of the full dataset — **largest single speedup** |
| `data.table` keyed assignment by reference | Avoids repeated deep-copy and logical scan of the full data frame |
| `fwrite()` instead of `write.csv()` | Orders-of-magnitude faster file writing |
| `rm()` + `gc()` inside loop | Keeps peak memory usage lower between iterations |
| No model retraining | Pre-trained `rf_models_per_year` are preserved and reused as-is |