 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (because the loop variable and the column share the same name), so `test_set` contains **all rows** every iteration instead of just one year's worth. This means predictions are run on the full dataset repeatedly, massively inflating computation time.

3. **Entire `joined_data` is held in memory as a data.frame/tibble:** With hundreds of thousands of rows and many predictor columns, this can be very large. Filtering it repeatedly inside a loop is inefficient.

4. **Row-by-row assignment into `pred_db$consolidated`:** Indexing a large data.frame with a logical vector inside a loop is slow, especially when the data.frame is large.

5. **`predict.randomForest` on a huge data.frame:** Even with a correct filter, predicting on hundreds of thousands of rows at once can spike memory because `randomForest::predict` builds an intermediate matrix of all tree predictions. This can be batched.

6. **Writing a massive CSV with `write.csv`:** For very large outputs, `data.table::fwrite` is dramatically faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read` call | Remove it |
| Variable shadowing (`year == year`) | Rename loop variable to `yr` |
| Repeated full-data filter | Pre-split `joined_data` by year using `split()` or `data.table` keying |
| Large intermediate objects | Convert to `data.table`; drop unneeded columns; use `gc()` |
| Memory spike during `predict()` | Batch predictions in chunks if a single year is still too large |
| Slow CSV write | Use `data.table::fwrite` |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)

# ── 1. Load only what is needed ──────────────────────────────────────────────
# Do NOT load the shapefile — it is unused in prediction.
# prep_data <- st_read(...)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Convert working tables to data.table for speed & memory efficiency ────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── 3. Key / index joined_data by year for fast subsetting ───────────────────
setkey(joined_data, year)

# ── 4. Optional: identify predictor columns once ─────────────────────────────
# If you know them, list them explicitly to avoid carrying extra columns
# into predict(). Example (adjust to your actual names):
# predictor_cols <- setdiff(names(joined_data), c("year", "cell_id", ...))

# ── 5. Batch‐size for chunked prediction (tune to your RAM) ─────────────────
BATCH_SIZE <- 50000L

# ── 6. Prediction loop — note the renamed iterator 'yr' ─────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Fast keyed subset (returns only rows for this year)
  test_set <- joined_data[.(yr)]

  n <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s — skipping.\n", yr))
    next
  }

  # Chunked prediction to cap peak memory
  if (n <= BATCH_SIZE) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    preds <- numeric(n)
    starts <- seq(1L, n, by = BATCH_SIZE)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + BATCH_SIZE - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
  }

  # Assign predictions back (data.table update‐by‐reference — no copy)
  pred_db[year == yr, consolidated := preds]

  # Free memory
  rm(test_set, preds)
  gc()
}

# ── 7. Write output efficiently ──────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Removed unused `st_read` | Saves potentially gigabytes of RAM |
| 2 | Renamed loop variable to `yr` | **Fixes the critical bug** where every year processed all rows |
| 3 | Converted to `data.table` + `setkey` | Subsetting by year is near-instant via binary search |
| 4 | Update-by-reference (`:=`) | Avoids copying the entire `pred_db` on each assignment |
| 5 | Chunked `predict()` calls | Caps peak memory; chunk size is tunable |
| 6 | `fwrite` instead of `write.csv` | Orders-of-magnitude faster for large files |
| 7 | `rm()` + `gc()` after each year | Frees memory between iterations |

No models are retrained; all existing `rf_models_per_year` objects are used as-is.