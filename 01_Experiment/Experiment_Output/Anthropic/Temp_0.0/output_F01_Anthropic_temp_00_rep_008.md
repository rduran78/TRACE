 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction workflow. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable shadowing bug:** `filter(year == year)` inside the loop uses the same name for the loop variable and the column, causing `dplyr` to compare the column to itself (always `TRUE`), so **every iteration predicts on the entire dataset** instead of a single year's subset. This is both a correctness bug and a massive performance problem.

4. **Full data frame in memory:** `joined_data` and `pred_db` are likely large `data.frame` or `tibble` objects. Subsetting with `dplyr::filter` creates full copies each iteration.

5. **Row-level assignment via logical indexing on a large data frame:** `pred_db$consolidated[pred_db$year == year]` scans the entire data frame every iteration.

6. **`predict.randomForest` on huge data:** With hundreds of thousands of rows and many trees, prediction is CPU- and memory-intensive. It can be batched.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries and data | Remove them to free memory |
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) |
| Repeated full-data copies | Use `data.table` for fast keyed subsetting |
| Large single-pass prediction | Batch prediction in chunks to limit peak memory |
| Slow row assignment | Use `data.table` set-by-reference with key |
| Output as CSV for huge data | Use `data.table::fwrite` (much faster than `write.csv`) |

**Constraint honored:** No models are retrained. The pre-trained `rf_models_per_year` list is used as-is.

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load pre-trained models and data ──────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year (named list), joined_data, pred_db

# ── Convert to data.table for speed and memory efficiency ─────────────────
setDT(joined_data)
setDT(pred_db)

# ── Pre-allocate the output column ────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Key both tables on 'year' for fast subsetting ─────────────────────────
setkey(joined_data, year)
setkey(pred_db, year)

# ── Batch-prediction helper (limits peak memory per predict call) ─────────
predict_in_batches <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  if (n <= batch_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (i in seq_along(starts)) {
    idx <- starts[i]:min(starts[i] + batch_size - 1L, n)
    preds[idx] <- predict(model, newdata = newdata[idx, ])
  }
  return(preds)
}

# ── Main prediction loop ─────────────────────────────────────────────────
available_years <- names(rf_models_per_year)

for (yr in unique(pred_db$year)) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% available_years) {
    # Fast keyed subset — no copy of the full table
    test_set <- joined_data[.(yr)]
    rf_model <- rf_models_per_year[[yr_char]]

    # Predict in manageable batches
    preds <- predict_in_batches(rf_model, test_set, batch_size = 50000L)

    # Set by reference — no copy
    pred_db[.(yr), consolidated := preds]

    # Free memory immediately
    rm(test_set, preds)
    gc()
  } else {
    cat(sprintf("  Warning: No model found for year %s\n", yr))
  }
}

# ── Write output efficiently ──────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Removed 12 unused library calls and the unused shapefile read | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable from `year` to `yr` | **Fixes the silent bug** where every year predicted on the full dataset |
| 3 | Converted to `data.table` with keys | Subsetting and assignment are orders of magnitude faster |
| 4 | Added `predict_in_batches()` | Caps peak memory during prediction; adjustable `batch_size` |
| 5 | Added `rm()` + `gc()` inside the loop | Releases memory between years |
| 6 | Replaced `write.csv` with `fwrite` | 5–10× faster file writing |
| 7 | Models are **not retrained** | Constraint preserved |