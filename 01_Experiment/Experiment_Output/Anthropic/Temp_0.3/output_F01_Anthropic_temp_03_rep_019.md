 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` equals the *entire* `joined_data` every iteration. This is both a correctness bug and a massive performance drain — every year predicts on the full dataset.

4. **Monolithic `predict()` on hundreds of thousands of rows:** Even after fixing the filter bug, calling `predict()` on a very large data frame in one shot can spike memory, especially with large Random Forest models (which store many trees).

5. **Keeping the full `joined_data` in memory:** If `joined_data` is large, it persists throughout the loop unnecessarily.

6. **Using `write.csv` for a large data frame:** This is slow; `data.table::fwrite()` is much faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` | Remove the `st_read()` call entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Large single `predict()` call | Chunk predictions into batches (e.g., 50,000 rows) |
| Full `joined_data` in memory | Subset per year, then discard; or convert to `data.table` |
| Slow CSV writing | Use `data.table::fwrite()` |
| General memory | Call `gc()` between years; use `data.table` instead of `tibble` |

---

## Optimized R Code

```r
# ── Load only what is needed ──
library(randomForest)
library(data.table)

# ── Load pre-trained models and associated data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed and lower memory ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Helper: chunked predict to limit peak memory ──
chunked_predict <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Prediction loop (with corrected filter) ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Correct subsetting — no variable masking
    test_set <- joined_data[year == yr]

    preds <- chunked_predict(rf_model, test_set, chunk_size = 50000L)

    pred_db[year == yr, consolidated := preds]

    # Free memory
    rm(test_set, preds)
    gc()
  } else {
    cat(sprintf("Warning: No model found for year %s\n", yr))
  }
}

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Why |
|---|---|---|
| 1 | Removed 12 unused libraries | Reduces memory by hundreds of MB |
| 2 | Removed unused `st_read()` call | Avoids loading a large shapefile into RAM |
| 3 | Fixed `filter(year == year)` → `joined_data[year == yr]` | **Critical bug fix**: was predicting on the entire dataset every iteration |
| 4 | Switched to `data.table` | Faster subsetting, lower memory overhead than `dplyr`/`tibble` |
| 5 | Added chunked prediction | Caps peak memory during `predict()` calls |
| 6 | Added `rm()` + `gc()` per iteration | Frees intermediate objects between years |
| 7 | Replaced `write.csv` with `fwrite` | Orders of magnitude faster for large outputs |

No models are retrained. All Random Forest objects from the `.RData` file are used as-is.