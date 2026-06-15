 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but is never used in the prediction workflow. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` always equals the entire `joined_data` dataframe. This means every iteration predicts on *all* rows, not just the target year, wasting time and producing incorrect results.

4. **Whole-dataframe column assignment in a loop:** Writing into `pred_db$consolidated[pred_db$year == year]` inside a loop over years triggers repeated full-column scans and copy-on-modify behavior in base R data frames.

5. **`predict.randomForest` on massive data:** With hundreds of thousands of rows and many trees, `predict()` is memory-intensive. Processing all years sequentially in one R session keeps all intermediate results in memory.

6. **`write.csv` on a large dataframe:** `write.csv` is slow; `data.table::fwrite` is dramatically faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused shapefile (`prep_data`) | Remove the `st_read()` call entirely |
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so `filter(year == yr)` works correctly |
| Slow loop assignment | Use `data.table` keyed joins or vectorized assignment |
| Large predict calls | Process year-by-year (already done), but now on correctly filtered (smaller) subsets |
| Slow CSV write | Use `data.table::fwrite()` |
| General memory | Call `gc()` after each year; optionally limit RF threads |

By fixing the shadowing bug alone, each `predict()` call operates on a single year's subset (potentially 10–50× fewer rows), which is the single largest performance gain.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)
library(dplyr)            # for filter(); much lighter than full tidyverse

# ── Load the pre-trained models and prediction database ──────────────
# (Do NOT load the shapefile — it is unused and wastes RAM)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for faster operations ─────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Prediction loop (note: loop variable is 'yr', not 'year') ───────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correctly filter to the current year only (fixes the shadowing bug)
  test_set <- joined_data[year == yr]

  model_key <- as.character(yr)

  if (model_key %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[model_key]]

    # Predict only on the (now correctly small) subset
    preds <- predict(rf_model, newdata = test_set)

    # Fast keyed assignment via data.table
    pred_db[year == yr, consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each iteration
  rm(test_set, preds)
  gc()
}

# ── Write output with fast fwrite instead of write.csv ───────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Change | Expected Impact |
|---|---|
| Remove `st_read()` of unused shapefile | Saves potentially **gigabytes** of RAM |
| Remove 10+ unused libraries | Reduces baseline memory by hundreds of MB |
| Fix `year == year` → `year == yr` | Each `predict()` runs on ~1/N-th of the data (N = number of years). **Largest single speedup.** |
| `data.table` instead of base data.frame | Faster subsetting and in-place column assignment; no copy-on-modify |
| `fwrite()` instead of `write.csv()` | ~5–10× faster file writing |
| `gc()` per iteration | Reclaims memory between years |

These changes preserve all trained Random Forest models exactly as-is (no retraining) and should make the workflow feasible on a standard personal computer.