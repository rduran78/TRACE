 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (since the loop variable and column share the same name), returning **all rows** instead of the intended subset. This means every iteration predicts on the entire dataset, massively inflating computation.

2. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop, wasting memory.

3. **Full data frame copy on every iteration:** Assigning predictions row-by-row into `pred_db$consolidated` with logical indexing inside a loop is inefficient for large data frames.

4. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive, especially with large forests and many predictors. Predicting in a single monolithic call per year may spike RAM.

5. **`write.csv` on a very large data frame** is slow; `data.table::fwrite` is dramatically faster.

6. **All libraries are loaded** even when most are unused in this script, increasing memory footprint.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing (`year == year`) | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Unused large spatial object (`prep_data`) | Remove the `st_read()` call entirely |
| Unnecessary libraries loaded | Load only `randomForest`, `dplyr`, and `data.table` |
| Large single-pass `predict()` | Predict in **chunks** (batches) to cap peak memory |
| Slow `write.csv` | Use `data.table::fwrite()` |
| Row-wise assignment in a loop | Pre-allocate a results list and bind once at the end, or use `data.table` keyed assignment |

The trained Random Forest models are **preserved untouched**; only the prediction loop is restructured.

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)   # predict method for RF models
library(dplyr)          # data manipulation
library(data.table)     # fast I/O and keyed operations

# ── Load pre-trained models and prediction database ───────────────────────
# (Do NOT load the large unused shapefile)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for fast keyed operations
setDT(pred_db)
setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# ── Chunked prediction function (caps peak memory) ───────────────────────
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0L))

  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Main prediction loop ─────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {                              # 'yr' avoids shadowing
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct filter: compare column to the loop variable 'yr'
  test_set <- joined_data[year == yr]

  if (nrow(test_set) == 0L) {
    cat(sprintf("  Warning: No rows in joined_data for year %s.\n", yr))
    next
  }

  # Predict in manageable chunks to limit peak memory
  preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

  # Assign back efficiently via data.table keyed update
  pred_db[year == yr, consolidated := preds]

  # Free memory immediately
  rm(test_set, preds)
  gc()
}

# ── Fast write ────────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done – predictions written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Why |
|---|---|---|
| 1 | Renamed loop variable to `yr` | **Fixes the critical filtering bug** that caused every year to predict on the full dataset |
| 2 | Removed `st_read()` of unused shapefile | Saves potentially gigabytes of RAM |
| 3 | Removed 10+ unused library calls | Reduces memory overhead and load time |
| 4 | Converted data frames to `data.table` | Faster subsetting and in-place column updates |
| 5 | Added `predict_in_chunks()` | Caps peak memory during prediction; chunk size is tunable |
| 6 | Added `rm()` + `gc()` after each year | Frees intermediate objects promptly |
| 7 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster for large outputs |

No models are retrained; the `rf_models_per_year` list is used as-is.