 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), returning the **entire dataset** every iteration instead of a single year's subset. This is both a correctness bug and a massive performance problem — every year's prediction runs on the full dataset.

4. **`predict.randomForest` on huge data frames:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows with many predictors in a single call can spike memory, especially if the Random Forest model has many trees and deep nodes.

5. **Writing a potentially enormous CSV at the end:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is far faster.

6. **No garbage collection or chunking:** The entire `joined_data` and `pred_db` live in memory simultaneously alongside all loaded models.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| Unused shapefile load | Remove `st_read()` call |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Full dataset predicted every iteration | Fix the filter so only one year's rows are predicted |
| Memory spike during `predict()` | Chunk large year-subsets into batches (e.g., 50,000 rows) |
| Slow CSV write | Use `data.table::fwrite()` |
| Accumulated memory pressure | Call `gc()` after each year |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and prediction data ───────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for speed (if not already)
setDT(pred_db)
setDT(joined_data)

# Initialise the output column
pred_db[, consolidated := NA_real_]

# ── Configurable chunk size ───────────────────────────────────────
CHUNK_SIZE <- 50000L   # rows per predict() call; tune to your RAM

# ── Prediction loop ──────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── Correct subsetting (no variable-masking bug) ──
  test_idx   <- which(joined_data$year == yr)
  pred_idx   <- which(pred_db$year == yr)
  test_set   <- joined_data[test_idx, ]

  n_rows     <- nrow(test_set)
  preds      <- numeric(n_rows)

  # ── Chunked prediction to limit peak memory ──
  n_chunks <- ceiling(n_rows / CHUNK_SIZE)
  for (ch in seq_len(n_chunks)) {
    row_start <- (ch - 1L) * CHUNK_SIZE + 1L
    row_end   <- min(ch * CHUNK_SIZE, n_rows)
    preds[row_start:row_end] <- predict(rf_model, newdata = test_set[row_start:row_end, ])
  }

  pred_db[pred_idx, consolidated := preds]

  # Free per-iteration objects and reclaim memory
  rm(rf_model, test_set, preds)
  gc()

  cat(paste0("  Done — ", n_rows, " rows predicted.\n"))
}

# ── Fast write ────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed 12 unused library calls and the `st_read()` call | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` and used direct `==` comparison | **Fixes the masking bug** — only the correct year is now subset |
| 3 | Switched to `data.table` for `pred_db` and `joined_data` | Faster subsetting and in-place column assignment |
| 4 | Added chunked `predict()` with configurable `CHUNK_SIZE` | Caps peak memory during prediction |
| 5 | Added `rm()` + `gc()` after each year | Returns memory to the OS between iterations |
| 6 | Replaced `write.csv` with `fwrite` | Typically 10–20× faster for large files |
| 7 | Pre-trained Random Forest models are **not retrained** | Requirement preserved |