 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but is never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geographic cells.

3. **Loading a monolithic `.RData` file:** The `load()` call likely brings in not only the trained models (`rf_models_per_year`) but also the full training data (`joined_data`, `pred_db`, and possibly other large objects), all of which sit in RAM simultaneously.

4. **Variable masking bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (due to name collision with the loop variable), so `test_set` is always the entire `joined_data` rather than a single-year subset. This means every iteration predicts on the full dataset — massively inflating computation time and memory use.

5. **Using a `data.frame` for row-wise assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a very large data frame inside a loop is slow due to repeated logical indexing and copy-on-modify semantics.

6. **`randomForest::predict` on huge data:** Even with the bug fixed, predicting hundreds of thousands of rows with a large Random Forest is memory-intensive because `predict.randomForest` can create large intermediate matrices.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consume memory | Load only what is needed |
| `prep_data` shapefile loaded but unused | Remove the `st_read()` call |
| Entire `.RData` loads all objects into RAM | Save models, `pred_db`, and `joined_data` as separate `.rds` files; load only what is needed, and free objects after use |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly |
| Slow row assignment on large data.frame | Use `data.table` for fast keyed joins/updates |
| Large single-year predictions may still be heavy | Predict in row-chunks within each year to cap peak memory |
| Writing a huge CSV is slow | Use `data.table::fwrite()` |

---

## Optimized R Code

```r
# ------------------------------------------------------------------
# 0.  Load only the libraries actually needed for prediction
# ------------------------------------------------------------------
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ------------------------------------------------------------------
# 1.  One-time preparatory step (run once, then comment out):
#     Split the monolithic .RData into lean .rds files.
#     This avoids loading everything into RAM every time.
# ------------------------------------------------------------------
# load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# saveRDS(rf_models_per_year, '/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
# saveRDS(pred_db,            '/Volumes/Toshi 1Tb/R_save_files/pred_db.rds')
# saveRDS(joined_data,        '/Volumes/Toshi 1Tb/R_save_files/joined_data.rds')
# rm(list = ls()); gc()

# ------------------------------------------------------------------
# 2.  Load only what is needed
# ------------------------------------------------------------------
rf_models_per_year <- readRDS('/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
pred_db            <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/pred_db.rds'))
joined_data        <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/joined_data.rds'))

# Ensure a key for fast subsetting
setkey(pred_db,     year)
setkey(joined_data, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# ------------------------------------------------------------------
# 3.  Chunked prediction loop
#     - 'yr' avoids the variable-masking bug (year == year)
#     - Predicting in chunks of CHUNK_SIZE rows caps peak memory
# ------------------------------------------------------------------
CHUNK_SIZE <- 50000L          # tune to your available RAM

available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in available_years) {

  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_num   <- as.numeric(yr)
  rf_model <- rf_models_per_year[[yr]]

  # Fast keyed subset (no copy of the whole table)
  test_set <- joined_data[.(yr_num)]
  n        <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s – skipped.\n", yr))
    next
  }

  # Predict in chunks to limit peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Fast update by reference (no copy)
  pred_db[.(yr_num), consolidated := preds]

  # Free per-year objects
  rm(test_set, preds)
  gc()

  cat(sprintf("  Done – %s rows predicted.\n", formatC(n, big.mark = ",")))
}

# ------------------------------------------------------------------
# 4.  Write output efficiently
# ------------------------------------------------------------------
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed 11 unused libraries and the unused `st_read()` call | Frees hundreds of MB of RAM |
| 2 | Split `.RData` → individual `.rds` files | Load only what you need; faster I/O |
| 3 | Renamed loop variable to `yr` | **Fixes the masking bug** — previously every year predicted on the *entire* dataset |
| 4 | Converted to `data.table` with keys | Subsetting and update-by-reference are orders of magnitude faster than base `data.frame` |
| 5 | Chunked `predict()` calls (default 50 000 rows) | Caps peak memory; chunk size is tunable |
| 6 | `gc()` after each year | Returns freed memory to the OS promptly |
| 7 | `fwrite()` instead of `write.csv()` | 5–10× faster CSV writing |

No models are retrained; only `predict()` is called on the existing `rf_models_per_year`.