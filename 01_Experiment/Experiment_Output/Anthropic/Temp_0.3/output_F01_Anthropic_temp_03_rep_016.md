 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), returning the **entire dataset** every iteration instead of a single year's subset. This is the most critical bug — it means every year's prediction runs on the full dataset, massively inflating computation time and memory.

4. **Using `tidyverse` filtering on large data:** `dplyr::filter()` on a large data frame is slower and more memory-intensive than `data.table` subsetting, which is already loaded.

5. **Writing the full `pred_db` as CSV in one shot:** If `pred_db` is very large, this can spike memory.

6. **Loading the entire `.RData` file into memory:** This loads `joined_data`, `rf_models_per_year`, and `pred_db` (and possibly other objects) all at once. If any are unnecessary, they waste memory.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read()` call entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable to avoid column-name collision |
| Large in-memory data | Convert `joined_data` and `pred_db` to `data.table` for fast, memory-efficient subsetting |
| Single large CSV write | Use `data.table::fwrite()` for faster I/O |
| Garbage collection | Call `gc()` after removing unneeded objects and after each year's prediction |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)

# ── Load trained models and prediction data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove any large objects loaded from .RData that are not needed.
# Keep only: joined_data, pred_db, rf_models_per_year
# If other objects were saved in that file, remove them explicitly, e.g.:
# rm(other_object_1, other_object_2)

# ── Convert to data.table for speed and memory efficiency ──
if (!is.data.table(joined_data)) setDT(joined_data)
if (!is.data.table(pred_db))     setDT(pred_db)

# ── Initialize the output column ──
pred_db[, consolidated := NA_real_]

# ── Free memory ──
gc()

# ── Predict year by year ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Subset only the rows for this year (data.table fast subset)
    test_set <- joined_data[year == yr]

    rf_model <- rf_models_per_year[[yr_char]]
    preds    <- predict(rf_model, newdata = test_set)

    pred_db[year == yr, consolidated := preds]

    # Free temporary objects
    rm(test_set, preds, rf_model)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output with fast I/O ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Impact |
|---|---|---|
| 1 | Removed 12 unused library calls | Lower memory footprint, faster startup |
| 2 | Removed unused `st_read()` call | Potentially saves GBs of RAM |
| 3 | Renamed loop variable to `yr` | **Fixes the critical bug** where every year processed the entire dataset |
| 4 | Converted to `data.table` with `setDT()` (in-place) | Faster subsetting with no copy overhead |
| 5 | Added `rm()` + `gc()` inside the loop | Frees memory between iterations |
| 6 | Replaced `write.csv()` with `fwrite()` | Significantly faster file writing |
| 7 | Preserved all trained RF models exactly as-is | No retraining performed |

The single most impactful fix is **#3**: the original `filter(year == year)` bug caused the full dataset to be predicted every iteration, multiplying both time and memory usage by the number of unique years.