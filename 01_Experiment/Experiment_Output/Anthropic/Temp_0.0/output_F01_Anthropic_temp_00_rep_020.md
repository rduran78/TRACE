 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — not just the target year. This massively inflates computation per iteration.

4. **Repeated full-dataframe assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` performs a logical scan of the entire dataframe on every iteration.

5. **`predict.randomForest` on huge dataframes:** With hundreds of thousands of rows and many predictors, a single `predict()` call can spike memory. This is unavoidable per year, but the bug in point 3 makes it far worse.

6. **`write.csv` on a large dataframe:** `write.csv` is slow; `data.table::fwrite` is dramatically faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` spatial object | Remove the `st_read()` call entirely |
| Variable masking in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) |
| Whole-dataframe scan per iteration | Convert `pred_db` to `data.table` and use keyed assignment |
| Slow CSV write | Use `data.table::fwrite()` |
| Memory pressure from large objects | Call `gc()` after loading the model file; predict year-by-year from `data.table` splits |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)       # kept only if joined_data / pred_db are tibbles

# ── Load pre-trained models and prediction data ───────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Free memory from any objects in the .RData that are not needed
# (adjust names if the file contains other large objects)
gc()

# ── Convert to data.table for speed ──────────────────────────────
setDT(pred_db)
setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# Key both tables on year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# ── Predict year-by-year ─────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Subset only the rows for this year (fast with data.table key)
    test_set <- joined_data[.(yr)]

    rf_model <- rf_models_per_year[[yr_char]]
    preds    <- predict(rf_model, newdata = test_set)

    # Assign predictions back (keyed join, no full-table scan)
    pred_db[.(yr), consolidated := preds]

    # Free per-iteration memory
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output efficiently ─────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Change | Benefit |
|---|---|
| Removed `st_read` of unused shapefile | Saves potentially **gigabytes** of RAM |
| Removed unused libraries | Reduces baseline memory and load time |
| Fixed `filter(year == year)` bug → `joined_data[.(yr)]` | Predicts on **one year's rows** instead of the entire dataset each iteration — massive speed and memory improvement |
| `data.table` keyed subsetting and assignment | O(log n) lookups instead of full logical scans |
| `gc()` after each iteration | Returns freed memory to the OS promptly |
| `fwrite()` instead of `write.csv()` | Typically **5–10×** faster for large files |

These changes preserve all trained Random Forest models exactly as-is (no retraining) and should make the workflow feasible on a standard personal computer.