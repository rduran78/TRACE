 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

There are several performance and correctness issues in this script:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), so every iteration loads the *entire* dataset instead of one year's subset. This is the single biggest source of unnecessary memory use and computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read` but never used in the prediction loop—this wastes memory loading a potentially large shapefile with geometry.

3. **Full Data Frame in Memory:** `joined_data` and `pred_db` are likely large `data.frame` or `tibble` objects. Subsetting and predicting on the full frame repeatedly is slow.

4. **Column-wise Assignment in a Loop:** Assigning predictions back into `pred_db$consolidated[pred_db$year == year]` each iteration forces repeated logical scans over the entire data frame.

5. **Unused Libraries:** Many loaded libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, `zoo`) are not needed for prediction, and loading them consumes memory and startup time.

6. **`write.csv` on a Huge Table:** Writing hundreds of thousands (or millions) of rows with `write.csv` is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable masking (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly, drastically reducing per-iteration data size |
| Unused shapefile in memory | Remove the `st_read` call (or defer it to when it's actually needed) |
| Unused libraries consuming RAM | Load only the libraries required for prediction |
| Slow `data.frame` operations | Convert `joined_data` and `pred_db` to `data.table` for fast keyed subsetting and update-by-reference |
| Slow CSV writing | Use `data.table::fwrite` |
| Optional: large predict batches | If a single year is still too large, chunk the `predict()` call |

These changes **preserve the trained Random Forest models exactly as-is**—no retraining occurs.

---

## Optimized Working R Code

```r
# ── Load only what is needed for prediction ──────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and associated data ──────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expects: rf_models_per_year (named list of RF models)
#           joined_data        (features used for prediction)
#           pred_db            (target table to receive predictions)

# ── Convert to data.table for speed and memory efficiency ────────────
setDT(joined_data)
setDT(pred_db)

# ── Initialise the output column ─────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Create keys for fast subsetting ──────────────────────────────────
setkey(joined_data, year)
setkey(pred_db, year)

# ── Predict year by year ─────────────────────────────────────────────
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {                       # <-- renamed loop var
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # ── Fast keyed subset (only this year's rows) ──────────────────────
  test_set <- joined_data[.(yr)]                     # data.table keyed lookup

  # ── Optional: predict in chunks if memory is still tight ───────────
  n        <- nrow(test_set)
  chunk_sz <- 50000L
  preds    <- numeric(n)

  rf_model <- rf_models_per_year[[yr_char]]

  if (n <= chunk_sz) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_sz)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + chunk_sz - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
  }

  # ── Update by reference (no copy of the whole table) ───────────────
  pred_db[.(yr), consolidated := preds]
}

# ── Free model objects if no longer needed ────────────────────────────
rm(rf_models_per_year, joined_data)
gc()

# ── Fast write ────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Change | Expected Benefit |
|---|---|
| Fix `year == year` → `yr` | **Largest gain:** each iteration now processes only one year instead of the entire dataset |
| `data.table` keyed subset | ~10–100× faster subsetting vs. `dplyr::filter` on large frames |
| Update by reference (`:=`) | Avoids copying the entire `pred_db` each iteration |
| Chunked `predict()` | Caps peak memory during prediction |
| Remove unused libraries/data | Frees hundreds of MB of RAM |
| `fwrite` vs. `write.csv` | ~5–10× faster file writing |

These changes should make the workflow feasible on a standard laptop with 8–16 GB of RAM without altering any trained model.