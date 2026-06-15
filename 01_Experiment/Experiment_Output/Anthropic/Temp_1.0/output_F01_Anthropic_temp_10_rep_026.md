 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `sf` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`prep_data`) that is never used** in the prediction logic — this wastes significant memory.

3. **The entire `joined_data` and `pred_db` dataframes are held in memory simultaneously**, likely loaded from the `.RData` file. With hundreds of thousands of rows and many predictors, this is very memory-intensive.

4. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), meaning every year's prediction uses the *entire* dataset rather than the correct subset. This makes every iteration process far more data than intended and produces incorrect results.

5. **`predict.randomForest` on hundreds of thousands of rows at once** can be memory-intensive because Random Forest prediction materializes predictions across all trees.

6. **Writing the full `pred_db` to CSV at the end** can be slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Change | Rationale |
|---|---|
| Remove unused libraries and the unused `prep_data` object | Free memory |
| Fix the variable shadowing bug (`year == year`) | Correctness + avoids processing the full dataset each iteration |
| Convert `joined_data` and `pred_db` to `data.table` | Faster subsetting and assignment |
| Predict in row-batches within each year | Caps peak memory from `predict()` |
| Remove large objects and call `gc()` after loading | Reclaim memory |
| Use `data.table::fwrite` instead of `write.csv` | Much faster I/O |
| Optionally drop unneeded columns from `joined_data` before prediction | Reduces memory footprint |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)

# ── Load trained models and associated data ──────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove the spatial file read entirely — it is unused in prediction.
# (Do NOT load prep_data.)

# ── Convert to data.table for speed and memory efficiency ────────────
setDT(pred_db)
setDT(joined_data)

# Free any other large objects brought in by the .RData file that are
# not needed (inspect ls() after load to identify them). Example:
# rm(object_not_needed); gc()
gc()

# ── Identify predictor columns used by the model (optional guard) ────
# sample_model <- rf_models_per_year[[names(rf_models_per_year)[1]]]
# needed_cols  <- c("year", sample_model$xNames)  # or rownames(importance(sample_model))
# joined_data  <- joined_data[, ..needed_cols]
# gc()

# ── Initialise output column ─────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Batch-size for predict() to limit peak memory ────────────────────
BATCH_SIZE <- 50000L          # tune to your machine; 50 k rows is safe

# ── Prediction loop — one year at a time, in batches ─────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── FIX: correctly subset by the current year value ────────────────
  test_idx   <- which(joined_data$year == yr)
  pred_idx   <- which(pred_db$year == yr)
  n          <- length(test_idx)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s — skipping.\n", yr))
    next
  }

  # ── Predict in batches to limit memory ─────────────────────────────
  preds <- numeric(n)
  starts <- seq(1L, n, by = BATCH_SIZE)

  for (s in starts) {
    e <- min(s + BATCH_SIZE - 1L, n)
    batch_rows <- test_idx[s:e]
    preds[s:e] <- predict(rf_model, newdata = joined_data[batch_rows, ])
  }

  pred_db[pred_idx, consolidated := preds]

  rm(rf_model, preds)
  gc()
}

# ── Write results (fwrite is ~5-10x faster than write.csv) ───────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

| # | What | Why |
|---|---|---|
| 1 | Removed 12 unused library calls and the unused `st_read` | Saves hundreds of MB of RAM |
| 2 | Fixed `filter(year == year)` → `joined_data$year == yr` | **Critical bug fix** — was selecting all rows every iteration |
| 3 | Converted to `data.table` | Faster subset and in-place `:=` assignment (no copy) |
| 4 | Batch prediction (`BATCH_SIZE`) | Caps peak memory of `predict.randomForest` |
| 5 | `rm()` + `gc()` after each year | Returns memory to OS promptly |
| 6 | `fwrite` instead of `write.csv` | Orders-of-magnitude faster file write |
| 7 | **No model retraining** | All existing `rf_models_per_year` objects are preserved and reused as-is |