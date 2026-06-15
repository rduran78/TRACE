 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading the entire spatial dataset unnecessarily.** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes significant memory for no benefit.

2. **Variable masking bug in `filter()`.** The line `filter(year == year)` compares the column `year` to itself (due to name collision with the loop variable), so it returns **all rows** instead of the subset for a single year. This means every iteration predicts on the entire dataset, massively increasing computation and producing incorrect results.

3. **Subsetting with `dplyr::filter()` on a potentially huge data frame each iteration.** Even once the bug is fixed, `filter()` on a tibble/data.frame for hundreds of thousands of rows inside a loop is slower than necessary.

4. **Row-by-row assignment into a large data frame with logical indexing (`pred_db$consolidated[pred_db$year == year]`) each iteration.** This triggers repeated full-column scans.

5. **`predict.randomForest` on a very large `newdata` can be memory-intensive**, especially if the forest has many trees and the data has many predictors. No chunking is performed.

6. **Keeping all loaded `.RData` objects in memory simultaneously** (e.g., `joined_data`, `pred_db`, `rf_models_per_year`, and possibly others) can exhaust RAM.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused `prep_data` spatial read | Remove it entirely |
| Variable masking bug (`year == year`) | Rename loop variable or use `.env$` pronoun |
| Slow subsetting & assignment in a `data.frame` | Convert `joined_data` and `pred_db` to `data.table` and use keyed subsetting |
| Large single-pass `predict()` call | Chunk prediction into batches (e.g., 50 000 rows) to cap peak memory |
| All objects in memory at once | Remove unneeded objects and call `gc()` after each year |
| Output as CSV for a huge table | Use `data.table::fwrite()` (much faster than `write.csv()`) |

**Key constraint honoured:** the trained Random Forest models (`rf_models_per_year`) are never retrained — only `predict()` is called.

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(data.table)
library(randomForest)   # needed for predict() on randomForest objects

# ── 1. Load saved workspace (models + data) ──────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove prep_data if it was loaded (unused); free memory immediately
if (exists("prep_data")) { rm(prep_data); gc() }

# ── 2. Convert to data.table for fast keyed operations ───────────────────
setDT(joined_data)
setDT(pred_db)

# Create keyed index on year for O(log n) subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# ── 3. Chunked prediction function (caps peak memory) ────────────────────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  # Pre-allocate result vector
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── 4. Prediction loop (one year at a time) ──────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # Keyed subset – fast, no copying of the full table
  test_set <- joined_data[.(yr)]

  rf_model <- rf_models_per_year[[yr_char]]

  # Chunked predict to limit memory spike
  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Direct keyed update (in-place, no full-column scan)
  pred_db[.(yr), consolidated := preds]

  # Free intermediate objects
  rm(test_set, preds)
  gc()

  cat(sprintf("  Done – %d cells predicted.\n", length(preds)))
}

# ── 5. Write output (fwrite is ~5-10× faster than write.csv) ─────────────
data.table::fwrite(pred_db, "RF_imputated_db.csv")

cat("All predictions written to RF_imputated_db.csv\n")
```

---

### Summary of Changes

| # | What changed | Why |
|---|---|---|
| 1 | Removed `st_read()` of unused shapefile | Saves potentially gigabytes of RAM |
| 2 | Removed unnecessary library loads | Faster startup, smaller footprint |
| 3 | Fixed `filter(year == year)` → keyed `joined_data[.(yr)]` | Eliminates the variable-masking bug **and** is faster |
| 4 | Switched to `data.table` with `setkey()` | Subsetting and in-place update are orders of magnitude faster |
| 5 | Added `predict_chunked()` | Prevents a single massive allocation when a year has hundreds of thousands of rows |
| 6 | `rm()` + `gc()` after each year | Releases memory between iterations |
| 7 | `fwrite()` instead of `write.csv()` | Much faster serialization for large tables |

These changes should make the script feasible on a personal computer with moderate RAM (8–16 GB) without retraining any model.