 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will equal the **entire** `joined_data` every iteration rather than the subset for that year. This means every iteration predicts on the full dataset — massively inflating computation time and producing incorrect results.

3. **Monolithic `predict()` calls on hundreds of thousands of rows:** Random Forest prediction on very large data frames with many predictors is memory-intensive. Predicting all cells at once for a year (especially when the bug above means *all* cells) can exhaust RAM.

4. **Use of `data.frame`-based operations:** `pred_db` is likely a large `data.frame`. Repeated subsetting and assignment via `pred_db$consolidated[pred_db$year == year]` is slow on large data frames compared to `data.table` keyed operations.

5. **Writing a massive CSV at the end:** `write.csv` is slow for large files; `data.table::fwrite` is significantly faster.

6. **Many unused libraries loaded:** Each loaded library consumes memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `prep_data` load | Remove `st_read()` call |
| Variable shadowing in `filter()` | Rename loop variable or use `.env$year` / `!!` to disambiguate |
| Entire dataset predicted every iteration | Correctly subset per year |
| Large single-pass `predict()` | Chunk predictions within each year to cap memory usage |
| Slow `data.frame` subsetting | Convert to `data.table` and use keyed assignment |
| Slow `write.csv` | Use `fwrite()` |
| Unused libraries | Remove them to reduce memory footprint |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)
library(dplyr)            # only if joined_data is a tibble; otherwise optional

# ── Load pre-trained models and prediction database ───────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: pred_db, joined_data, rf_models_per_year

# ── Convert to data.table for speed ──────────────────────────────────────
setDT(pred_db)
setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# Set key for fast subsetting
setkey(pred_db,   year)
setkey(joined_data, year)

# ── Tunable chunk size (rows per predict() call) ─────────────────────────
# Adjust based on available RAM.  Smaller = less peak memory.
CHUNK_SIZE <- 50000L

# ── Prediction loop ──────────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  model_key <- as.character(yr)
  if (!(model_key %in% names(rf_models_per_year))) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[model_key]]

  # Correct subset: use the renamed loop variable `yr`
  test_set <- joined_data[year == yr]

  n <- nrow(test_set)
  if (n == 0L) {
    cat(paste0("  No observations for year ", yr, " — skipping.\n"))
    next
  }

  # Chunked prediction to cap memory usage
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Fast keyed assignment back into pred_db
  pred_db[year == yr, consolidated := preds]

  # Free memory after each year
  rm(test_set, preds)
  gc()
}

# ── Write output quickly ─────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

1. **Bug fix** — renamed loop variable to `yr` so `joined_data[year == yr]` correctly subsets a single year instead of returning the entire table.
2. **Chunked `predict()`** — processes `CHUNK_SIZE` rows at a time, keeping peak memory manageable on a personal computer.
3. **`data.table` throughout** — keyed subsetting and `:=` assignment are orders of magnitude faster than base `data.frame` bracket assignment on large tables.
4. **Removed unused `st_read()`** — avoids loading a potentially multi-GB shapefile that is never referenced.
5. **Removed unused libraries** — reduces baseline memory consumption.
6. **`fwrite()` instead of `write.csv()`** — typically 5–10× faster for large files.
7. **Explicit `gc()` per year** — returns freed memory to the OS between iterations.

These changes preserve all pre-trained Random Forest models exactly as-is; no retraining occurs.