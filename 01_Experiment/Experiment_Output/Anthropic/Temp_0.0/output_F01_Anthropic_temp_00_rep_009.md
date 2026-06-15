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
| Remove unused libraries and the `st_read` call | Free memory immediately |
| Fix the variable masking bug in `filter()` | Prevent predicting on the entire dataset every year |
| Convert `pred_db` and `joined_data` to `data.table` | Faster subsetting, joins, and assignment by reference |
| Subset predictor columns before calling `predict()` | Reduce memory footprint passed to the model |
| Use `data.table::fwrite` instead of `write.csv` | Much faster I/O for large files |
| Optionally call `gc()` after each year | Encourage R to release memory between iterations |

**No models are retrained.** The `rf_models_per_year` list is loaded and used as-is.

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load pre-trained models and data ──────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year (named list), pred_db, joined_data

# ── Convert to data.table for speed and memory efficiency ─────────────────
setDT(pred_db)
setDT(joined_data)

# ── Identify the predictor columns the RF models expect ───────────────────
# (take them from any one of the stored models)
example_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(example_model$importance)

# ── Initialise the output column ──────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Predict year by year ──────────────────────────────────────────────────
years_to_predict <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in years_to_predict) {
  cat("Predicting for year", yr, "... ")

  # Correct subsetting — no variable-masking bug
  test_set <- joined_data[year == as.numeric(yr), ..predictor_vars]

  rf_model <- rf_models_per_year[[yr]]
  preds    <- predict(rf_model, newdata = test_set)

  # Assign by reference (fast, no copy)
  pred_db[year == as.numeric(yr), consolidated := preds]

  cat("done. Rows predicted:", length(preds), "\n")

  # Free memory between iterations
  rm(test_set, preds)
  gc()
}

# ── Report any years without a model ──────────────────────────────────────
missing <- setdiff(as.character(unique(pred_db$year)),
                   names(rf_models_per_year))
if (length(missing) > 0) {
  cat("Warning: No model found for year(s):", paste(missing, collapse = ", "), "\n")
}

# ── Write output (fwrite is ~5-10x faster than write.csv) ────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

### Summary of Key Changes

| Original Problem | Fix Applied |
|---|---|
| `filter(year == year)` selects **all** rows every loop | Use `data.table` syntax: `joined_data[year == as.numeric(yr), ...]` — unambiguous |
| ~15 libraries loaded, most unused | Only `data.table` and `randomForest` loaded |
| Large shapefile loaded but never used | `st_read` call removed entirely |
| Slow `write.csv` | Replaced with `data.table::fwrite` |
| Slow logical-index assignment on `data.frame` | `data.table` assignment by reference (`:=`) |
| Full-width data passed to `predict()` | Only `predictor_vars` columns passed |
| No memory cleanup between years | `rm()` + `gc()` after each iteration |

These changes should make the workflow feasible on a standard personal computer without retraining any models.