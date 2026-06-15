 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (because the loop variable and the column share the same name). This means `test_set` contains **all rows** every iteration rather than a single year's subset, causing massively redundant computation.

3. **Entire `joined_data` held in memory:** If `joined_data` is a large `sf` or `data.frame` with hundreds of thousands of rows × many columns, keeping it all in memory alongside `pred_db` and the RF models is expensive.

4. **`predict.randomForest` on huge data frames:** Predicting on hundreds of thousands of rows at once with a large Random Forest can spike memory because the function internally allocates matrices proportional to `n_rows × n_trees`.

5. **Writing a massive CSV in one shot:** `write.csv` on a very large data frame can be slow and memory-hungry compared to `data.table::fwrite`.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused `prep_data` load | Remove it |
| Variable shadowing (`year == year`) | Rename the loop variable (e.g., `yr`) |
| Full dataset in memory | Convert to `data.table`; subset in place |
| Large single-pass prediction | Predict in chunks (batches) within each year |
| Slow CSV write | Use `data.table::fwrite` |
| All models loaded at once | Optionally load/unload per year, but since the `.RData` bundles them, at minimum ensure nothing else is wasting memory |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load saved models and prediction database ─────────────────────────────
# This loads rf_models_per_year, pred_db, joined_data (and possibly others)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove any large objects that were loaded but are not needed for prediction
# Adjust the list below based on what the .RData file actually contains
if (exists("prep_data"))   rm(prep_data)
gc()

# ── Convert to data.table for speed and memory efficiency ─────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Configurable batch size (tune to your available RAM) ──────────────────
BATCH_SIZE <- 50000L   # number of rows predicted at a time

# ── Prediction loop ──────────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct subsetting: use the loop variable, not the column name
  test_idx <- which(joined_data$year == yr)

  if (length(test_idx) == 0L) {
    cat(sprintf("  Warning: No rows in joined_data for year %s.\n", yr))
    next
  }

  # Predict in batches to limit peak memory
  n_batches <- ceiling(length(test_idx) / BATCH_SIZE)
  preds     <- numeric(length(test_idx))

  for (b in seq_len(n_batches)) {
    start <- (b - 1L) * BATCH_SIZE + 1L
    end   <- min(b * BATCH_SIZE, length(test_idx))
    batch_rows <- test_idx[start:end]
    preds[start:end] <- predict(rf_model, newdata = joined_data[batch_rows, ])
  }

  # Write predictions back into pred_db
  pred_rows <- which(pred_db$year == yr)
  pred_db[pred_rows, consolidated := preds]

  # Free memory after each year
  rm(rf_model, preds)
  gc()
}

# ── Write output efficiently ──────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Why |
|---|---|---|
| 1 | Removed `st_read` of unused shapefile | Saves significant memory |
| 2 | Removed all unused `library()` calls | Faster startup, less memory |
| 3 | Renamed loop variable to `yr` | **Fixes the silent bug** where `filter(year == year)` always returns all rows |
| 4 | Converted to `data.table` | Faster subsetting, lower memory overhead |
| 5 | Added batched prediction | Caps peak memory during `predict()` |
| 6 | Added `rm()` + `gc()` after each year | Frees memory between iterations |
| 7 | Replaced `write.csv` with `fwrite` | Dramatically faster I/O |
| 8 | No model retraining | All original RF models are preserved and reused as-is |

Adjust `BATCH_SIZE` downward if you still encounter memory pressure, or upward if you have RAM to spare and want fewer loop iterations.