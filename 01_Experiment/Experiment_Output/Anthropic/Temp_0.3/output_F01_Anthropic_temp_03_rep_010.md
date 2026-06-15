 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction loop. This wastes significant memory, especially if the shapefile is large.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not work as intended. The loop variable `year` shadows the column name `year`, so the filter matches *all* rows every iteration rather than just the target year. This means the `predict()` call processes the entire dataset on every iteration, massively inflating computation time.

4. **Predicting on the full dataset repeatedly:** Because of the bug above, every year's prediction pass sends hundreds of thousands (or millions) of unnecessary rows through the Random Forest model.

5. **Using a `data.frame` for row-level assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a very large data frame inside a loop is slow due to repeated indexing and copy-on-modify semantics.

6. **`randomForest::predict` is single-threaded and memory-heavy on large data:** Sending all rows at once (or too many rows due to the bug) can spike memory.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the unused `prep_data` object | Free memory |
| Fix the variable masking bug by renaming the loop variable | Ensure only the correct year's rows are predicted |
| Convert to `data.table` for fast subsetting and assignment | Avoid copy-on-modify overhead |
| Optionally batch large year-groups into chunks | Cap peak memory during `predict()` |
| Use `fwrite` instead of `write.csv` | Much faster I/O for large files |
| Call `gc()` between years | Release memory promptly |

The trained Random Forest models are **not retrained** — only the prediction loop is optimized.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data is a tibble / uses dplyr structures

# ── Load the pre-trained models and associated data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load the shapefile — it is unused in prediction.
# prep_data <- st_read(...)   # REMOVED to save memory

# ── Convert to data.table for fast indexed operations ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Prediction loop (bug-fixed and optimized) ──
# Rename loop variable to avoid masking the column name "year"
years_to_predict <- unique(pred_db$year)

# Optional: set a chunk size to cap memory during predict()
CHUNK_SIZE <- 50000L    # adjust based on available RAM

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!(yr_char %in% names(rf_models_per_year))) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct filter: use the renamed loop variable 'yr'
  test_set <- joined_data[year == yr]

  n <- nrow(test_set)

  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, " — skipping.\n"))
    next
  }

  # Predict in chunks to limit peak memory
  if (n <= CHUNK_SIZE) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    preds <- numeric(n)
    starts <- seq(1L, n, by = CHUNK_SIZE)
    for (i in seq_along(starts)) {
      idx_start <- starts[i]
      idx_end   <- min(idx_start + CHUNK_SIZE - 1L, n)
      preds[idx_start:idx_end] <- predict(rf_model,
                                           newdata = test_set[idx_start:idx_end])
    }
  }

  # Fast indexed assignment via data.table
  pred_db[year == yr, consolidated := preds]

  # Free memory before next iteration
  rm(test_set, preds)
  gc()
}

# ── Write output efficiently ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Effect |
|---|--------|--------|
| 1 | Removed 10+ unused library calls and the unused shapefile read | Reduces memory footprint by potentially gigabytes |
| 2 | Renamed loop variable from `year` to `yr` | **Fixes the critical filter bug** — previously every year predicted on the *entire* dataset |
| 3 | Converted `pred_db` and `joined_data` to `data.table` | Faster subsetting (`[year == yr]`) and in-place column assignment (no deep copies) |
| 4 | Added chunked prediction (`CHUNK_SIZE`) | Caps peak memory during `predict()` for years with hundreds of thousands of rows |
| 5 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster file writing |
| 6 | Added `gc()` after each year | Promptly reclaims memory |
| 7 | Models are **untouched** — no retraining | Preserves the original `rf_models_per_year` exactly as loaded |

These changes together should make the prediction process feasible on a standard personal computer with moderate RAM (8–16 GB).