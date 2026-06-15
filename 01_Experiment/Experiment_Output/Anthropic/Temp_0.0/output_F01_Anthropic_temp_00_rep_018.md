 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — massively inflating computation time and producing incorrect predictions.

4. **Using `tidyverse` filtering in a loop over large data:** `dplyr::filter()` on a large data frame inside a loop is slower than vectorized or `data.table` subsetting.

5. **`predict.randomForest` on the full dataset:** When `test_set` accidentally contains all rows (due to the bug above), prediction is called on the entire dataset for every year — an O(n × k) explosion.

6. **`pred_db` may be a large data frame with many unused columns:** Writing the entire object to CSV may be slow and produce a very large file.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the unused `st_read()` call | Free significant memory |
| Fix the variable masking bug (`year == year`) | Correct predictions and dramatically reduce per-iteration data size |
| Convert to `data.table` for fast subsetting | `data.table` subset-by-reference is much faster than `dplyr::filter` for large data |
| Pre-select only the columns needed for prediction | Reduces memory footprint of each subset passed to `predict()` |
| Use `set()` or vectorized assignment in `data.table` | Avoids repeated copy-on-modify overhead |
| Optionally, call `gc()` between years | Helps reclaim memory on constrained machines |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load the saved models and associated data ─────────────────────
# This file should contain: rf_models_per_year, joined_data, pred_db
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast operations ─────────────────────
setDT(joined_data)
setDT(pred_db)

# ── Identify predictor columns used by the first available model ──
sample_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(sample_model$importance)

# ── Pre-allocate the output column ────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Get the unique years that have a trained model ────────────────
years_to_predict <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

# ── Prediction loop (fixed masking bug, minimal memory) ───────────
for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_num <- as.numeric(yr)

  # Subset only the rows for this year and only the needed columns
  test_set <- joined_data[year == yr_num, ..predictor_vars]

  # Predict
  preds <- predict(rf_models_per_year[[yr]], newdata = test_set)

  # Assign predictions back (vectorised, in-place)
  pred_db[year == yr_num, consolidated := preds]

  # Free memory
  rm(test_set, preds)
  gc()
}

# ── Report any years without a model ──────────────────────────────
missing_years <- setdiff(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)
if (length(missing_years) > 0) {
  cat("Warning: No model found for year(s):", paste(missing_years, collapse = ", "), "\n")
}

# ── Write results (only key columns to reduce file size) ──────────
fwrite(pred_db[, .(year, consolidated)], "RF_imputed_db.csv")
# If you need all columns:
# fwrite(pred_db, "RF_imputed_db.csv")

cat("Done.\n")
```

### Key Changes Summary

- **Bug fix:** Renamed the loop variable to `yr` so `year == yr_num` correctly subsets a single year instead of matching every row.
- **Memory:** Removed `st_read()` and all unused libraries, subset only predictor columns, and call `gc()` each iteration.
- **Speed:** `data.table` subsetting and `fwrite()` are orders of magnitude faster than `dplyr::filter()` and `write.csv()` on large data.
- **Models untouched:** No retraining; the existing `rf_models_per_year` objects are used as-is.