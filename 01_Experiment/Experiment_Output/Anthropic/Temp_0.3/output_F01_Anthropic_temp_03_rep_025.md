 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — not just the target year. This multiplies computation time by the number of unique years.

4. **Using a full `tidyverse` data.frame pipeline:** For hundreds of thousands of rows, `dplyr::filter()` on a standard data frame is slower than keyed `data.table` subsetting.

5. **Writing the entire data frame with `write.csv`:** For large data, `write.csv` is slow; `data.table::fwrite()` is significantly faster.

6. **`predict.randomForest` on massive data:** This is inherently memory-intensive because it passes all data through every tree. Chunked prediction can keep peak memory lower.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused `library()` calls and the `st_read()` call | Frees hundreds of MB of RAM |
| Fix the variable masking bug (`year == year`) | Reduces each iteration's data to the correct subset — massive speedup |
| Convert to `data.table` and use keyed subsetting | Faster row filtering than `dplyr::filter` |
| Predict in chunks within each year | Caps peak memory usage |
| Replace `write.csv` with `fwrite` | Much faster I/O for large files |
| Use `gc()` between years | Reclaims memory promptly |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load the pre-trained models and prediction database ───────────────────
# (Assumes the .RData file contains: rf_models_per_year, joined_data, pred_db)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed ──────────────────────────────────────
setDT(pred_db)
setDT(joined_data)

# Key both tables on year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Initialise the output column
pred_db[, consolidated := NA_real_]

# ── Chunked prediction function (controls peak memory) ───────────────────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# ── Main prediction loop ─────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct subsetting — avoids the year == year masking bug
  test_set <- joined_data[.(yr)]

  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  pred_db[.(yr), consolidated := preds]

  # Free memory after each year
  rm(test_set, preds)
  gc()
}

# ── Fast write ────────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

| Original Problem | Fix Applied |
|---|---|
| `filter(year == year)` selects **all** rows every iteration | Renamed loop variable to `yr`; used `data.table` keyed subset `joined_data[.(yr)]` |
| ~15 unused libraries consuming RAM | Reduced to 2 (`randomForest`, `data.table`) |
| Unused `st_read()` of large shapefile | Removed entirely |
| `predict()` on full-year block may spike memory | `predict_chunked()` processes 50 000 rows at a time |
| Slow `write.csv` | Replaced with `fwrite` |
| No explicit garbage collection | `gc()` called after each year |

These changes preserve the trained Random Forest models exactly as-is (no retraining) and should make the workflow feasible on a standard personal computer.