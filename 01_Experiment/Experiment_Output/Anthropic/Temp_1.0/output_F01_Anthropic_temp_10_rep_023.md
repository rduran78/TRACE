 # Diagnosis and Optimization of Cell-Level GDP Prediction Script

---

## Diagnosis

There are several performance and correctness issues in this script:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always TRUE), so every iteration loads the *entire* dataset into `test_set` instead of just one year's slice. This means *every* prediction call sends hundreds of thousands of unnecessary rows to `predict()`.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used, consuming significant memory for a large shapefile.

3. **Heavy Library Loading:** Many libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are loaded but not used in the prediction phase, adding overhead.

4. **Inefficient Row-Indexing on a Data Frame:** Repeated subsetting of `pred_db` with logical conditions (`pred_db$year == year`) inside a loop is slow on large data frames.

5. **`predict.randomForest` on Huge Batches:** Even correctly filtered, predicting hundreds of thousands of rows at once can spike memory. Chunked prediction would be safer.

6. **CSV Output:** `write.csv()` on a very large data frame is slow; `data.table::fwrite()` is dramatically faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable masking bug | Rename loop variable to `yr` and filter with `.env$yr` or `!!yr` |
| Unused shapefile in memory | Remove the `st_read()` call |
| Unused libraries | Load only what is needed |
| Slow row-indexing | Convert to `data.table`, key by year, use `:=` assignment |
| Memory spikes during `predict()` | Predict in chunks within each year |
| Slow CSV write | Use `data.table::fwrite()` |

---

## Optimized Working Code

```r
# ── Load only necessary libraries ───────────────────────────────
library(data.table)
library(randomForest)   # needed for predict() dispatch
library(tidyverse)      # only if joined_data is a tibble; otherwise removable

# ── Load saved models and data ──────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expects: pred_db, joined_data, rf_models_per_year

# ── Convert to data.table for speed ────────────────────────────
setDT(pred_db)
setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# Key both tables by year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# ── Chunk-size for predict() to limit memory ───────────────────
CHUNK_SIZE <- 50000L   # adjust to your available RAM

# ── Prediction loop ────────────────────────────────────────────
years_to_predict <- intersect(
  unique(pred_db$year),
  as.numeric(names(rf_models_per_year))
)

for (yr in years_to_predict) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  rf_model  <- rf_models_per_year[[as.character(yr)]]

  # Correct filter: use the loop variable, not the column

  test_rows <- joined_data[.(yr)]            # fast keyed subset
  n         <- nrow(test_rows)

  if (n == 0L) {
    cat(sprintf("  No test rows for year %s – skipping.\n", yr))
    next
  }

  # Predict in chunks to avoid memory spikes
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_rows[s:e, ])
  }

  # Assign back via keyed join (fast)
  pred_db[.(yr), consolidated := preds]

  # Free memory after each year
  rm(test_rows, preds)
  gc()
}

# Warn about years with no model
missing_years <- setdiff(unique(pred_db$year), years_to_predict)
if (length(missing_years) > 0) {
  cat("Warning: No model found for year(s):", paste(missing_years, collapse = ", "), "\n")
}

# ── Write output (fwrite is ~5-10x faster than write.csv) ──────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Aspect | Before | After |
|---|---|---|
| Rows sent to `predict()` per year | Entire dataset (bug) | Only that year's rows |
| Memory from unused shapefile | Hundreds of MB+ | 0 |
| Subsetting speed | Base R logical vector scan | `data.table` keyed O(log n) |
| Peak prediction memory | Entire year at once | Controlled via `CHUNK_SIZE` |
| Write speed | `write.csv` | `fwrite` (~5–10× faster) |
| Correctness | Every year gets same (wrong) predictions | Each year gets its own model's predictions |

The single largest performance win comes from **fixing the variable-masking bug**, which alone reduces per-iteration work by a factor equal to the number of distinct years. The `data.table` conversion and chunked prediction then make the corrected workflow feasible on a standard personal computer.