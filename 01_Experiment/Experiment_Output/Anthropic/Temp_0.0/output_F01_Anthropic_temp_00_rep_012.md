 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), returning the **entire dataset** every iteration instead of a single year's subset. This is the most critical bug — it means every year's prediction runs on the full dataset, massively inflating computation time and memory.

4. **Using `dplyr::filter` on a potentially large data.frame** is slower than `data.table` subsetting for hundreds of thousands of rows.

5. **`predict.randomForest` on huge data in one call** can spike memory. Processing in chunks can help.

6. **Writing the full `pred_db` with `write.csv`** is slow for large data; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and objects | Remove `st_read()` call and unneeded libraries |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Large single-pass prediction | Optionally chunk predictions within each year |
| Slow CSV write | Use `data.table::fwrite` |
| Memory pressure | Call `gc()` after each year; convert `pred_db` to `data.table` |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)   # predict method for RF models
library(data.table)     # fast subsetting and writing

# ── Load pre-trained models and prediction database ───────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for speed and lower memory overhead ─────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Prediction loop ──────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {

    rf_model <- rf_models_per_year[[yr_char]]

    # Correct subsetting — no variable-masking bug
    test_idx  <- which(joined_data$year == yr)
    test_set  <- joined_data[test_idx]

    pred_idx  <- which(pred_db$year == yr)

    # ── Optional chunked prediction to limit peak memory ──
    chunk_size <- 50000L
    n          <- nrow(test_set)
    preds      <- numeric(n)

    starts <- seq(1L, n, by = chunk_size)
    for (s in starts) {
      e <- min(s + chunk_size - 1L, n)
      preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
    }

    set(pred_db, i = pred_idx, j = "consolidated", value = preds)

    # Free memory from this iteration
    rm(test_set, preds)
    gc()

  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Fast write ────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Removed 11 unused `library()` calls and the `st_read()` call | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` to fix the `filter(year == year)` masking bug | Correct predictions; avoids predicting on the full dataset every iteration |
| 3 | Switched from `dplyr` to `data.table` subsetting | Faster row selection on large tables |
| 4 | Added chunked `predict()` (configurable `chunk_size`) | Caps peak memory during prediction |
| 5 | Replaced `write.csv` with `fwrite` | ~5–10× faster file writing |
| 6 | Added `rm()` + `gc()` per iteration | Returns memory to the OS between years |

No models are retrained; only the prediction and I/O pipeline is optimized.