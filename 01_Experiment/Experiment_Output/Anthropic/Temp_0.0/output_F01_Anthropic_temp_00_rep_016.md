 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` equals the **entire** `joined_data` every iteration. This is both a correctness bug and a massive performance problem — every year predicts on the full dataset.

4. **Monolithic `predict()` on hundreds of thousands of rows:** Even after fixing the filter bug, calling `predict()` on a very large data frame in one shot can spike memory, especially with large Random Forest models (which store many trees).

5. **Using `data.frame` operations:** `pred_db` is likely a large data frame; indexed row assignment (`pred_db$consolidated[pred_db$year == year]`) is slow on large data frames.

6. **Writing a massive CSV at the end:** `write.csv()` is slow for large files compared to `data.table::fwrite()`.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| `prep_data` loaded but unused | Remove the `st_read()` call |
| Variable masking in `filter()` | Use `.env$year` or rename the loop variable |
| Entire dataset predicted every year | Fix the filter so only the relevant year is predicted |
| Memory spike on large `predict()` calls | Predict in chunks (batches) within each year |
| Slow `data.frame` assignment | Convert `pred_db` to `data.table` for fast indexed assignment |
| Slow `write.csv()` | Use `data.table::fwrite()` |
| Keeping all year-models in memory | Optionally load/unload models one at a time if memory is tight |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and prediction database ───────────────────────
# (Do NOT load prep_data — it is unused and wastes memory)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed ──────────────────────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Prediction settings ──────────────────────────────────────────────────
CHUNK_SIZE <- 50000L          # rows per chunk; tune to your RAM
years      <- unique(pred_db$year)

# ── Predict year-by-year, in chunks ──────────────────────────────────────
for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── Correct filter: use the loop variable, not the column ──────────

  test_idx <- which(joined_data$year == yr)

  if (length(test_idx) == 0L) {
    cat(sprintf("  Warning: No rows in joined_data for year %s.\n", yr))
    next
  }

  # ── Predict in chunks to limit peak memory ─────────────────────────
  n_chunks  <- ceiling(length(test_idx) / CHUNK_SIZE)
  preds_all <- numeric(length(test_idx))

  for (i in seq_len(n_chunks)) {
    start <- (i - 1L) * CHUNK_SIZE + 1L
    end   <- min(i * CHUNK_SIZE, length(test_idx))
    chunk <- joined_data[test_idx[start:end], ]
    preds_all[start:end] <- predict(rf_model, newdata = chunk)
    if (i %% 10 == 0 || i == n_chunks) {
      cat(sprintf("    chunk %d / %d done\n", i, n_chunks))
    }
  }

  # ── Write predictions back (data.table fast indexed assignment) ────
  pred_rows <- which(pred_db$year == yr)
  set(pred_db, i = pred_rows, j = "consolidated", value = preds_all)

  # ── Free the model copy to reclaim memory ──────────────────────────
  rm(rf_model, chunk, preds_all)
  gc()
}

# ── Fast write ────────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

- **Bug fix:** Replaced `filter(year == year)` with `which(joined_data$year == yr)` to correctly subset each year.
- **Memory reduction:** Removed all unused library loads and the unused `st_read()` call.
- **Chunked prediction:** `predict()` is called on batches of `CHUNK_SIZE` rows instead of the full year at once, capping peak memory.
- **`data.table` throughout:** Fast indexed assignment via `set()` and fast CSV output via `fwrite()`.
- **Explicit garbage collection:** `rm()` + `gc()` after each year frees the model copy and intermediate objects.
- **Models are preserved:** No retraining occurs; `rf_models_per_year` is loaded and used as-is.