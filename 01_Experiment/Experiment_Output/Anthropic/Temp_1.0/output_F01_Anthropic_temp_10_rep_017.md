 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (due to the loop variable sharing the same name), effectively returning *all* rows instead of the intended subset. This means every year's prediction runs on the entire dataset — massively inflating computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full Data Frame Copies:** Using `dplyr::filter` on a potentially very large `joined_data` tibble/data.frame each iteration is slow. `data.table` subsetting would be far more efficient.

4. **Row-by-Row Assignment to a Data Frame:** Assigning predictions into `pred_db$consolidated` via logical indexing on a large data.frame each iteration is inefficient due to R's copy-on-modify semantics.

5. **`predict.randomForest` on Huge Data:** Even with a correctly filtered subset, `predict()` on hundreds of thousands of rows with many predictors can be memory-intensive. Processing in chunks helps.

6. **Writing a massive CSV:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) |
| Unused `prep_data` load | Remove it |
| Slow subsetting | Convert to `data.table` and key on `year` |
| Copy-on-modify assignment | Pre-allocate a results list and `rbindlist` at the end, or assign by reference with `data.table` |
| Memory pressure from `predict()` | Predict in chunks within each year |
| Slow CSV write | Use `fwrite()` |
| Unused libraries | Remove to reduce load time/memory |

---

## Optimized R Code

```r
# ── Load only necessary libraries ────────────────────────────────
library(randomForest)
library(data.table)

# ── Load saved models ────────────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert core data to data.table and key by year ──────────────
setDT(pred_db)
setDT(joined_data)
setkey(joined_data, year)

# ── Pre-allocate the output column ───────────────────────────────
pred_db[, consolidated := NA_real_]
setkey(pred_db, year)

# ── Chunk size for predict() to limit peak memory ────────────────
CHUNK_SIZE <- 50000L  # tune based on available RAM

# ── Prediction loop ──────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!(yr_char %in% names(rf_models_per_year))) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correctly subset the data for this year
  test_set <- joined_data[.(yr)]          # fast keyed lookup
  n        <- nrow(test_set)

  if (n == 0L) {
    cat(paste0("  Warning: No test data for year ", yr, " — skipping.\n"))
    next
  }

  # Predict in chunks to control memory usage
  preds <- numeric(n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Assign by reference — no copy of the entire data.table
  pred_db[.(yr), consolidated := preds]

  # Free memory immediately
  rm(test_set, preds, rf_model)
  gc()
}

# ── Write output with fast fwrite ────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Why |
|---|---|---|
| 1 | Renamed loop variable to `yr` | **Fixes the critical filter bug** that caused every year to process all rows |
| 2 | Removed `st_read` of unused shapefile | Saves potentially gigabytes of RAM |
| 3 | Converted to `data.table` with `setkey(year)` | O(log n) subsetting vs. full-scan `dplyr::filter` |
| 4 | Used `:=` assignment by reference | Avoids R's copy-on-modify overhead on large objects |
| 5 | Chunked `predict()` calls | Caps peak memory; chunk size is tunable |
| 6 | Explicit `rm()` + `gc()` per iteration | Releases memory between years |
| 7 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster I/O |
| 8 | Removed unnecessary library loads | Reduces startup time and baseline memory |

No models are retrained. The existing `rf_models_per_year` list is used as-is.