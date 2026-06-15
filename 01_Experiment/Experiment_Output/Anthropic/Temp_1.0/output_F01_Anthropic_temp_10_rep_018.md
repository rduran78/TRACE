 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the provided script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This consumes memory for no reason.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (because the loop variable and the column share the same name), so `test_set` is always the *entire* `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on the full dataset — massively increasing both time and memory usage.

3. **Monolithic `predict()` on a huge dataframe:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows at once with a large Random Forest model can spike memory because `randomForest::predict` builds a matrix of predictions across all trees internally.

4. **Using a `data.frame` for cell-wise assignment:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a very large data.frame is slow. `data.table` set-by-reference operations are far faster.

5. **Writing output with `write.csv`:** `write.csv` is slow for large files. `data.table::fwrite` is dramatically faster.

6. **All loaded libraries:** Several libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are not needed for prediction and consume memory at load time.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `prep_data` spatial read | Remove the `st_read()` call |
| Unused libraries | Load only what is needed for prediction |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$` pronoun |
| Memory spike from full-data `predict()` | Process each year separately (which the loop intends) and optionally chunk large years |
| Slow data.frame assignment | Convert `pred_db` to `data.table` and use set-by-reference |
| Slow CSV write | Use `fwrite()` |
| Optional: garbage collection | Call `gc()` after each year to free tree-prediction intermediates |

---

## Optimized R Code

```r
# ── Only load libraries needed for prediction ──
library(randomForest)
library(data.table)
library(tidyverse)      # kept for joined_data compatibility; could be replaced

# ── Load the saved workspace (models + data) ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast by-reference operations ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Prediction loop (fixed filter + chunked option) ──
CHUNK_SIZE <- 50000L   # tune to your RAM; set to Inf to disable chunking

years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  model_key <- as.character(yr)

  if (!(model_key %in% names(rf_models_per_year))) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[model_key]]

  # ---- Correctly subset the current year (fix the shadowing bug) ----
  test_idx <- which(joined_data$year == yr)
  test_set <- joined_data[test_idx]

  n <- nrow(test_set)
  pred_idx <- which(pred_db$year == yr)

  if (n == 0L || length(pred_idx) == 0L) {
    cat(paste0("  No rows for year ", yr, " — skipping.\n"))
    next
  }

  # ---- Chunked prediction to limit peak memory ----
  preds <- numeric(n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # ---- Write results by reference (fast, no copy) ----
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)

  # ---- Free intermediate objects and reclaim memory ----
  rm(test_set, preds)
  gc()

  cat(paste0("  Done — ", n, " cells predicted.\n"))
}

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

### Summary of Key Changes

1. **Removed** `st_read()` and all unused library loads → frees hundreds of MB.
2. **Renamed** the loop variable to `yr` → fixes the silent bug where every year predicted on the entire dataset.
3. **Converted** to `data.table` and used `set()` → avoids costly whole-column copies on each assignment.
4. **Added chunked `predict()`** → caps peak memory per call; `CHUNK_SIZE` is tunable (50 000 rows is a safe default).
5. **Added `gc()`** per iteration → returns memory from Random Forest internal prediction matrices.
6. **Replaced `write.csv`** with `fwrite()` → typically 10–20× faster for large files.

No models are retrained; the existing `rf_models_per_year` list is used as-is.