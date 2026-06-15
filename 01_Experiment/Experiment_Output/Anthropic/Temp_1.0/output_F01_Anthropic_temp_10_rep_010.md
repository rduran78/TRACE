 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used, consuming memory for no reason.
2. **Heavy libraries loaded unnecessarily:** Libraries like `blockCV`, `LongituRF`, `spdep`, `terra`, `plm`, `fixest`, etc., are loaded but not used in the prediction loop, inflating memory footprint.
3. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (the column name shadows the loop variable), meaning `test_set` always equals the entire `joined_data` dataframe. This causes every year's prediction to run on the full dataset — massively inflating both computation time and memory.
4. **Using `data.frame` instead of `data.table`:** For hundreds of thousands of rows and many columns, `data.frame` subsetting and assignment are slow.
5. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year]` rescans the full dataframe on every iteration.
6. **Writing a potentially huge CSV with `write.csv`:** This is slower than `fwrite()` from `data.table`.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read()` call | Remove it entirely |
| Unnecessary libraries | Load only what is needed (`randomForest`, `data.table`) |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` / `data.table` syntax |
| Slow subsetting & assignment | Convert to `data.table` and use keyed operations |
| Slow CSV write | Use `data.table::fwrite()` |
| Optional: memory pressure from large `.RData` | After extracting what's needed, remove unneeded objects with `rm()` and call `gc()` |

The trained Random Forest models are **preserved exactly as-is** — no retraining occurs.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)

# ── Load the saved model workspace ───────────────────────────────────
# This should contain at least: rf_models_per_year, joined_data, pred_db
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Remove any large objects that are not needed (adjust as needed) ──
# For example, if prep_data or other objects were saved in the .RData:
if (exists("prep_data")) { rm(prep_data) }
gc()

# ── Convert to data.table for speed ─────────────────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Pre-allocate the output column ───────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Key both tables on year for fast subsetting ──────────────────────
setkey(pred_db, year)
setkey(joined_data, year)

# ── Prediction loop (models are NOT retrained) ──────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Fast keyed subset — no full-table scan
    test_set <- joined_data[.(yr)]

    rf_model <- rf_models_per_year[[yr_char]]

    # Predict and assign back via keyed join
    pred_db[.(yr), consolidated := predict(rf_model, newdata = test_set)]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each year's predictions
  gc()
}

# ── Write output efficiently ─────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed `st_read()` and 12 unused library calls | Saves hundreds of MB of RAM |
| 2 | Renamed loop variable from `year` to `yr` | **Fixes the critical bug** where `filter(year == year)` always returned all rows |
| 3 | Converted `pred_db` and `joined_data` to `data.table` with keys | Subsetting and assignment go from O(n) scan to near O(1) keyed lookup |
| 4 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster file writing |
| 5 | Added `rm()` / `gc()` calls | Frees memory between years and at startup |
| 6 | Models are loaded and used as-is | No retraining; original models fully preserved |

These changes should make the script feasible on a personal computer with modest RAM (8–16 GB) even for datasets with hundreds of thousands of cells per year.