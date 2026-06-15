 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), returning the **entire dataset** every iteration instead of a single year's subset. This massively inflates computation per iteration.

4. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- predict(...)` performs repeated logical indexing on a potentially very large data frame, which is slow.

5. **Using `data.frame` instead of `data.table`:** For hundreds of thousands of rows, `data.table` keyed operations are far more efficient for subsetting and assignment.

6. **`predict.randomForest` on huge subsets:** Because of the masking bug, the model predicts on the full dataset each year. Even after fixing the bug, prediction on hundreds of thousands of rows can be memory-intensive if many trees and variables are involved.

7. **Writing a massive CSV at the end:** `write.csv` is slow for large data; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the unused `st_read` call | Free memory |
| Fix the variable masking bug in `filter()` | Predict only on the correct year's subset, drastically reducing per-iteration work |
| Convert `pred_db` and `joined_data` to `data.table` with keys | Fast subsetting and update-by-reference |
| Use `data.table` update-by-reference (`:=`) | Avoids copying the entire data frame on each assignment |
| Use `fwrite` instead of `write.csv` | Much faster I/O |
| Optionally, garbage-collect after loading the model file | Reclaim memory from objects loaded but not needed |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data was built with dplyr pipelines

# ── Load pre-trained models and associated data ───────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load the shapefile — it is unused in prediction.
# prep_data <- st_read(...)   # REMOVED to save memory

# ── Convert to data.table for speed ──────────────────────────────
setDT(pred_db)
setDT(joined_data)

# Add keyed index for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Pre-allocate the output column (NA, numeric to match predictions)
pred_db[, consolidated := NA_real_]

# ── Free memory from any other large objects loaded from .RData ──
# Keep only rf_models_per_year, pred_db, and joined_data.
# If other large objects were loaded, remove them:
objs_to_keep <- c("rf_models_per_year", "pred_db", "joined_data")
rm(list = setdiff(ls(), objs_to_keep))
invisible(gc())

# ── Prediction loop (fixed and optimized) ────────────────────────
years <- sort(unique(pred_db$year))

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # FIXED: use .() or explicit object to avoid column-masking bug
    test_set <- joined_data[.(yr)]          # data.table keyed subset

    preds <- predict(rf_model, newdata = test_set)

    # Update by reference — no copy of the whole table
    pred_db[.(yr), consolidated := preds]

    # Free per-iteration memory
    rm(test_set, preds)
    invisible(gc())
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output (fwrite is ~5-10× faster than write.csv) ────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| Original Problem | Fix Applied |
|---|---|
| `filter(year == year)` selects **all** rows (masking bug) | Replaced with `data.table` keyed subset `joined_data[.(yr)]` using an unambiguous loop variable `yr` |
| Unused `st_read` loads a large shapefile into memory | Removed entirely |
| 14 libraries loaded; most unused | Reduced to 3 |
| Slow `data.frame` indexing in loop | Converted to `data.table` with `setkey` and update-by-reference (`:=`) |
| `write.csv` on large output | Replaced with `fwrite` |
| No garbage collection | Added `gc()` after cleanup and inside the loop |

These changes preserve all trained Random Forest models exactly as-is (no retraining) and should make the prediction workflow feasible on a standard personal computer.