 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on the full dataset — massively inflating computation time and producing incorrect results.

3. **Predicting on the entire dataset at once (per iteration):** Even after fixing the bug, calling `predict()` on hundreds of thousands of rows with a large Random Forest model can spike memory usage because the model must route every row through every tree simultaneously.

4. **Using a base `data.frame` for column assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` repeatedly on a very large data.frame is slow due to repeated indexing and potential memory copying.

5. **Unnecessary libraries loaded:** Many libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are not used in the prediction step and consume memory.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused spatial data load | Remove the `st_read()` call |
| Unused libraries | Load only `randomForest`, `data.table`, and `dplyr` (or just `data.table`) |
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so the filter works correctly |
| Memory spike during `predict()` | Predict in chunks (batches) within each year |
| Slow data.frame assignment | Convert `pred_db` to a `data.table` and use keyed or index-based update |
| Output efficiency | Use `fwrite()` instead of `write.csv()` |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and associated data ───────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for speed ──────────────────────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Batch-prediction settings ────────────────────────────────────────────
BATCH_SIZE <- 50000L          # rows per chunk; tune to your RAM

# ── Predict year by year ─────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correctly subset to the current year (avoids the shadowing bug)
  test_set <- joined_data[year == yr]
  n        <- nrow(test_set)

  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, ".\n"))
    next
  }

  # Predict in chunks to limit peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = BATCH_SIZE)

  for (s in starts) {
    e <- min(s + BATCH_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Fast indexed update
  pred_db[year == yr, consolidated := preds]

  # Free memory after each year
  rm(test_set, preds, rf_model)
  gc()
}

# ── Write output efficiently ─────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

- **Removed** `st_read()` and all unused library calls → frees significant RAM.
- **Fixed the filter bug** by renaming the loop variable to `yr`, so `joined_data[year == yr]` correctly subsets a single year.
- **Chunked prediction** (`BATCH_SIZE`) prevents a single `predict()` call from allocating a massive matrix for all rows × all trees simultaneously.
- **`data.table`** replaces base `data.frame` for fast subsetting and in-place column updates (no deep copies).
- **`gc()`** after each year reclaims memory promptly.
- **`fwrite()`** replaces `write.csv()` for much faster file output.

No models are retrained; only the prediction loop is restructured.