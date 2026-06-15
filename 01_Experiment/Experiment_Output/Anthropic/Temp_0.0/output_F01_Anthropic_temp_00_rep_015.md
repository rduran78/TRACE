 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` equals the **entire** `joined_data` every iteration. This is both a correctness bug and a massive performance problem — every year predicts on the full dataset.

4. **Monolithic `predict()` on hundreds of thousands of rows:** Even after fixing the filter bug, calling `predict()` on a very large data frame in one shot can spike memory, especially with large Random Forest models (which store many trees).

5. **Using `data.frame` operations:** `pred_db` is likely a large data frame; indexed row assignment (`pred_db$consolidated[pred_db$year == year] <- ...`) is slow on large data frames.

6. **Writing a massive CSV at the end:** `write.csv()` is slow for large files.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read()` call entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) |
| Memory spike during `predict()` | Predict in chunks (batches) to cap peak memory |
| Slow data.frame indexing | Convert `pred_db` to `data.table` for fast indexed assignment |
| Slow CSV writing | Use `data.table::fwrite()` instead of `write.csv()` |
| Optional: parallelism | Not pursued here to keep memory low on a personal machine |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)
library(dplyr)          # only for the initial filter; could also use data.table

# ── Load the pre-trained models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for fast indexed operations ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Set a chunk size that fits comfortably in RAM ──
CHUNK_SIZE <- 50000L   # adjust based on available memory

# ── Prediction loop ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correct filter: compare column to the loop variable, not to itself
  test_set <- joined_data[year == yr]

  model_key <- as.character(yr)

  if (!model_key %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[model_key]]
  n <- nrow(test_set)

  # ── Predict in chunks to limit peak memory ──
  preds <- numeric(n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # ── Fast indexed assignment via data.table ──
  pred_db[year == yr, consolidated := preds]

  # ── Free memory after each year ──
  rm(test_set, preds)
  gc()
}

# ── Optionally free the models if no longer needed ──
rm(rf_models_per_year, joined_data)
gc()

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Removed 11 unused library calls and the unused `st_read()` | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` to fix the `filter(year == year)` self-comparison bug | **Correctness fix** — previously predicted on the entire dataset every iteration |
| 3 | Switched `pred_db` and `joined_data` to `data.table` | Much faster subsetting and assignment |
| 4 | Predict in configurable chunks (`CHUNK_SIZE`) | Caps peak memory; avoids out-of-memory crashes |
| 5 | Explicit `rm()` + `gc()` after each year | Returns memory to the OS promptly |
| 6 | Replaced `write.csv()` with `fwrite()` | Typically 5–10× faster for large files |

No models are retrained; only the prediction and I/O pipeline is optimized.