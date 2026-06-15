 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on the full dataset — massively inflating computation time and producing incorrect results.

3. **Predicting on the entire dataset at once (per iteration):** Even after fixing the bug, calling `predict()` on hundreds of thousands of rows with a large Random Forest model can spike memory usage because the model must route every row through every tree simultaneously.

4. **Using a base `data.frame` for column assignment in a loop:** Repeated assignment to `pred_db$consolidated[pred_db$year == year]` on a very large data.frame triggers repeated memory copies.

5. **Writing a massive CSV at the end:** `write.csv()` on a very large data.frame is slow; `data.table::fwrite()` is significantly faster.

6. **Many heavy libraries loaded but unused:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, and `sf` are loaded but not needed for prediction, consuming memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and data | Remove `st_read()` call and all unused `library()` calls. |
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so it doesn't collide with the column name. |
| Memory spike during `predict()` | Predict in chunks (batches) within each year to cap peak memory. |
| Slow column assignment | Convert `pred_db` to a `data.table` and use keyed or indexed updates. |
| Slow CSV write | Use `data.table::fwrite()`. |
| Garbage collection | Call `gc()` after each year to free memory promptly. |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data / pred_db rely on tibble ops

# ── Load the saved workspace (models + data) ─────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast indexed operations ─────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# Create an index on the year column for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# ── Chunked prediction function ──────────────────────────────────────────
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))

  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  for (i in seq_along(starts)) {
    idx_start <- starts[i]
    idx_end   <- min(idx_start + chunk_size - 1L, n)
    preds[idx_start:idx_end] <- predict(model, newdata = newdata[idx_start:idx_end, ])
  }
  return(preds)
}

# ── Main prediction loop (note: loop var is 'yr', not 'year') ────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Correct subsetting — no variable shadowing
    test_set <- joined_data[year == yr]

    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

    pred_db[year == yr, consolidated := preds]

    # Free memory after each year
    rm(test_set, preds)
    gc()
  } else {
    cat(sprintf("Warning: No model found for year %s\n", yr))
  }
}

# ── Fast write ────────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

- **Removed** the unused `st_read()` call and 11 unnecessary library imports → frees hundreds of MB of RAM.
- **Fixed the shadowing bug** (`year == year` → loop variable renamed to `yr`) → each year now correctly subsets its own rows, dramatically reducing per-iteration work.
- **Converted to `data.table`** with `setkey()` → subsetting and in-place update via `:=` avoid full-copy overhead.
- **Chunked `predict()`** (default 50 000 rows per batch) → caps peak memory so the process stays feasible on a normal machine.
- **Replaced `write.csv` with `fwrite`** → typically 5–10× faster for large files.
- **Added `gc()` per iteration** → promptly reclaims memory between years.

No models are retrained; only the prediction and data-handling pipeline is optimized.