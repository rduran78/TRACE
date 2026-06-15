 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable name shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (because the loop variable `year` shadows the column name), returning *all* rows instead of the intended subset. This means every iteration predicts on the entire dataset — massively wasteful and incorrect.

2. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used, consuming memory for nothing.

3. **Entire `joined_data` held in memory as a data.frame/tibble:** With hundreds of thousands of rows and many predictors, `dplyr::filter()` on a large tibble each iteration is slower than necessary.

4. **Column assignment via logical indexing on a large data.frame (`pred_db$consolidated[pred_db$year == year]`):** This is slow for large data.frames because R copies the entire column on each assignment.

5. **`predict.randomForest` on a huge test set in one call:** For very large test sets, `predict()` can spike memory because the Random Forest must route every observation through every tree simultaneously.

6. **Many heavy libraries loaded but unused:** `blockCV`, `zoo`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra` — none are needed for prediction, yet they consume memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consume memory | Remove all unnecessary `library()` calls |
| Unused `prep_data` spatial object | Remove the `st_read()` call entirely |
| Variable shadowing bug (`year == year`) | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Slow subsetting and assignment on data.frame | Convert `pred_db` and `joined_data` to `data.table` for fast keyed subsetting and update-by-reference |
| Potential memory spike in `predict()` | Predict in chunks (batches) to cap peak memory |
| Writing a huge CSV at the end | Use `data.table::fwrite()` for much faster I/O |

The trained Random Forest models (`rf_models_per_year`) are **preserved exactly as-is** — no retraining occurs.

---

## Optimized R Code

```r
# ── Only the libraries actually needed for prediction ──
library(data.table)
library(randomForest)

# ── Load the saved models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for speed and memory-efficient operations ──
setDT(pred_db)
setDT(joined_data)

# ── Initialise the output column ──
pred_db[, consolidated := NA_real_]

# ── Set keys for fast subsetting ──
setkey(pred_db, year)
setkey(joined_data, year)

# ── Batch-size for chunked prediction (tune to your RAM) ──
BATCH_SIZE <- 50000L

# ── Predict year by year ──
unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)


  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Fast keyed subset (no copy of the whole table)
  test_set <- joined_data[.(yr)]
  n <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s — skipping.\n", yr))
    next
  }

  # ── Chunked prediction to cap peak memory ──
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = BATCH_SIZE)

  for (s in starts) {
    e <- min(s + BATCH_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # ── Update by reference (no column copy) ──
  pred_db[.(yr), consolidated := preds]

  # ── Free memory immediately ──
  rm(test_set, preds, rf_model)
  gc()
}

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Why |
|---|---|---|
| 1 | Removed 11 unused `library()` calls and the unused `st_read()` | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` | Fixes the `filter(year == year)` shadowing bug that caused incorrect (and bloated) subsets |
| 3 | Converted to `data.table` with keys | Subsetting and assignment by reference are orders of magnitude faster than base/dplyr on large frames |
| 4 | Chunked `predict()` calls (`BATCH_SIZE`) | Prevents a single massive allocation inside `predict.randomForest`; tune the batch size to your available RAM |
| 5 | Explicit `rm()` + `gc()` after each year | Releases memory from the previous year's test set before the next iteration |
| 6 | `fwrite()` instead of `write.csv()` | Typically 5–10× faster for large files |

No models are retrained; `rf_models_per_year` is loaded and used read-only.