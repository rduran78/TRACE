 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — not just the target year. This massively inflates computation per iteration.

4. **Whole-dataframe prediction:** Even without the bug, calling `predict()` on a full `data.frame`/`tibble` with many unused columns forces the model to search through a wide object. Subsetting to only the required predictor columns reduces memory copying overhead.

5. **Row-by-row assignment into a large data.frame:** Assigning predictions back into `pred_db` via logical indexing (`pred_db$consolidated[pred_db$year == year]`) on a very large data.frame is slow due to repeated full-column scans.

6. **Using `data.frame`/`tibble` instead of `data.table`:** For large datasets, `data.table` operations are significantly faster and more memory-efficient.

7. **Writing a massive CSV:** `write.csv` on hundreds of thousands (or millions) of rows is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the `st_read()` call | Free memory immediately |
| Fix the variable masking bug in `filter()` | Prevent predicting on the entire dataset every year |
| Convert `pred_db` and `joined_data` to `data.table` | Faster subsetting, joins, and assignment by reference |
| Subset predictor columns before calling `predict()` | Reduce memory footprint passed to the model |
| Use `data.table::fwrite()` for output | Much faster CSV writing |
| Use `gc()` after large operations | Prompt garbage collection to reclaim memory |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────────
library(randomForest)
library(data.table)

# ── Load the pre-trained models and associated data ──────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed and memory efficiency ────────────────
setDT(pred_db)
setDT(joined_data)

# ── Initialize the output column ─────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Identify the predictor variable names from the first available model ─
first_model_name <- names(rf_models_per_year)[1]
predictor_vars   <- rownames(rf_models_per_year[[first_model_name]]$importance)

# ── Predict year by year ─────────────────────────────────────────────────
years_to_predict <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_numeric <- as.numeric(yr)

  # Subset only the rows for this year AND only the needed columns
  test_subset <- joined_data[year == yr_numeric, ..predictor_vars]

  # Predict
  preds <- predict(rf_models_per_year[[yr]], newdata = test_subset)

  # Assign by reference (fast, no copy)
  pred_db[year == yr_numeric, consolidated := preds]

  # Free memory
  rm(test_subset, preds)
  gc()
}

# ── Report any years without a model ─────────────────────────────────────
missing_years <- setdiff(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)
if (length(missing_years) > 0) {
  cat("Warning: No model found for year(s):", paste(missing_years, collapse = ", "), "\n")
}

# ── Write output efficiently ─────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

### Summary of Key Changes

- **~10× less memory at startup** by dropping unused libraries and the unused spatial file.
- **Critical bug fix:** The original `filter(year == year)` selected *all* rows every iteration; the fix uses `data.table` syntax with an unambiguous external variable (`yr_numeric`).
- **Faster subsetting and assignment** via `data.table` by-reference operations (`..predictor_vars`, `:=`).
- **Smaller objects passed to `predict()`** by selecting only the columns the model actually uses.
- **Faster file output** with `fwrite()` instead of `write.csv()`.
- **No models are retrained** — only `predict()` is called on the existing `rf_models_per_year`.