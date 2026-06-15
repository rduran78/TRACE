 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading**: Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary data loading**: `prep_data` is read from a shapefile (`st_read`) but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **The `.RData` file likely loads multiple large objects into memory** (e.g., `joined_data`, `pred_db`, `rf_models_per_year`, and possibly training data), many of which may not all be needed simultaneously.

4. **Variable masking bug**: `filter(year == year)` inside the `for` loop does not work as intended. The loop variable `year` shadows the column name `year`, so the filter effectively evaluates to `TRUE` for every row, meaning `test_set` equals the entire `joined_data` every iteration. This causes the `predict()` call to score *all* rows every year instead of just one year's subset, massively increasing computation time and producing incorrect results.

5. **`predict.randomForest` on hundreds of thousands of rows with many trees is memory-intensive**: Random Forest prediction allocates intermediate matrices proportional to `n_rows × n_trees`. Doing this in one call on an unnecessarily large dataset (due to the bug) compounds the problem.

6. **Writing the full `pred_db` as CSV at the end**: If `pred_db` is very large, this is a slow final step, but is secondary to the above issues.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Remove unused libraries | Reduce memory footprint and load time |
| 2 | Remove the unused `st_read` call | Avoid loading a large shapefile into RAM |
| 3 | Fix the variable masking bug in `filter()` | Ensure only the current year's rows are predicted, drastically reducing per-iteration workload |
| 4 | Use `data.table` for subsetting instead of `dplyr::filter` | Faster subsetting on large data |
| 5 | Predict in chunks if a single year is still too large | Keeps peak memory bounded |
| 6 | Use `fwrite` instead of `write.csv` | Much faster for large data frames |
| 7 | Explicitly call `gc()` after each year | Free memory between iterations |
| 8 | Optionally, load only the needed objects from `.RData` | Avoid loading unnecessary training artifacts |

---

## Optimized R Code

```r
# ── Only load what is needed ──────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load saved models and prediction data ─────────────────────────
# This loads rf_models_per_year, joined_data, pred_db (and possibly others)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast operations ─────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Optional: set key for fast subsetting ─────────────────────────
setkey(joined_data, year)
setkey(pred_db, year)

# ── Chunk-based prediction function (controls peak memory) ────────
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
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

# ── Main prediction loop ──────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correct subsetting: use the loop variable yr, not the column name
    test_set <- joined_data[year == yr]

    rf_model <- rf_models_per_year[[yr_char]]
    preds    <- predict_in_chunks(rf_model, newdata = test_set, chunk_size = 50000L)

    pred_db[year == yr, consolidated := preds]

    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Improvements

- **Bug fix** (`filter(year == year)` → `joined_data[year == yr]`): This alone likely reduces per-iteration data size by an order of magnitude (from all years to one year), which is the single largest performance gain.
- **Memory reduction**: Dropping unused libraries and the shapefile read frees hundreds of MB to GB of RAM.
- **Chunked prediction**: Prevents a single `predict()` call from allocating an excessively large intermediate matrix if a single year still has very many rows.
- **`data.table` + `fwrite`**: Faster subsetting and I/O compared to `dplyr` and `write.csv`.
- **Trained models are fully preserved**; no retraining occurs.