 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary heavy library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — not just the target year. This massively inflates computation per iteration.

4. **Whole-dataframe prediction:** Even without the bug, calling `predict()` on a full `tidyverse` tibble/data.frame with many unused columns is slower than passing only the required predictor columns.

5. **Repeated column assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` row-by-row-group on a large data.frame is inefficient. Using `data.table` would be faster.

6. **Writing a massive CSV:** `write.csv` on hundreds of thousands (or millions) of rows is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Remove unused libraries and the unused `st_read()` call | Free memory immediately |
| 2 | Fix the variable masking bug (`year == year`) | Prevent predicting on the entire dataset every iteration |
| 3 | Convert working data to `data.table` | Faster subsetting and assignment |
| 4 | Subset only the predictor columns needed by the RF model before calling `predict()` | Reduce memory passed to `predict()` |
| 5 | Use `data.table::fwrite()` instead of `write.csv()` | Much faster I/O |
| 6 | Optionally, call `gc()` after each year to reclaim memory | Helps on RAM-constrained machines |

**No models are retrained.** All `rf_models_per_year` objects are preserved as-is.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(data.table)
library(randomForest)

# ── Load the saved models and associated prediction data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This loads: rf_models_per_year, pred_db, joined_data (and possibly others)

# ── Convert to data.table for speed ──
setDT(pred_db)
setDT(joined_data)

# ── Identify predictor columns from the first available model ──
first_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(first_model$importance)

# ── Initialize the output column ──
pred_db[, consolidated := NA_real_]

# ── Get the years that have a trained model ──
model_years <- names(rf_models_per_year)
unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% model_years) {
    # Subset only the rows for this year and only the needed predictor columns
    # This fixes the original variable-masking bug (year == year)
    test_set <- joined_data[year == yr, ..predictor_vars]

    rf_model <- rf_models_per_year[[yr_char]]

    # Predict and assign back efficiently via data.table
    pred_db[year == yr, consolidated := predict(rf_model, newdata = test_set)]

    # Free memory
    rm(test_set)
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

- **Memory savings:** Removed ~10 unused library loads and the large unused spatial shapefile read — potentially freeing gigabytes of RAM.
- **Correctness fix:** Replaced `filter(year == year)` (which selected *all* rows every iteration) with `joined_data[year == yr, ...]`, correctly subsetting by year.
- **Speed gains:** Switched to `data.table` for subsetting and column assignment; passed only the required predictor columns to `predict()`; replaced `write.csv` with `fwrite`.
- **Models untouched:** All pre-trained Random Forest models in `rf_models_per_year` are used as-is with no retraining.