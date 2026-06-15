 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `scales`, `stringi`, and `spdep` are not used in the prediction loop. Loading them wastes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries can consume gigabytes of RAM.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly because the column name `year` and the loop variable `year` share the same name. This causes the filter to return **all rows** every iteration instead of a single year's subset, massively inflating the prediction workload.

4. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` triggers repeated full-column scans and copy-on-modify behavior in base R data frames on every iteration.

5. **`predict.randomForest` on a huge data frame:** When the (buggy) filter returns the entire dataset, prediction is run on hundreds of thousands of unnecessary rows, consuming excessive CPU and memory.

6. **Writing a potentially huge CSV at the end:** If `pred_db` is very large, `write.csv` is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` spatial object | Remove the `st_read()` call entirely |
| Variable masking in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) |
| Repeated full-column indexing | Convert `pred_db` to `data.table` and use keyed assignment by reference |
| Large single-call `predict()` | Optionally chunk predictions within a year if memory is still tight |
| Slow `write.csv` | Use `data.table::fwrite` |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────────
library(randomForest)
library(tidyverse)
library(data.table)

# ── Load the saved models and associated data ────────────────────────────
# This file should contain: rf_models_per_year, pred_db, joined_data
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast by-reference operations ───────────────
setDT(pred_db)
setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# Key both tables on year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# ── Prediction loop ─────────────────────────────────────────────────────
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Correct subsetting — no variable-masking bug
    test_set <- joined_data[.(yr)]

    if (nrow(test_set) == 0L) {
      cat(paste0("  Warning: No test data for year ", yr, "\n"))
      next
    }

    # ── Optional chunked prediction to limit peak memory ──
    chunk_size <- 50000L
    n <- nrow(test_set)

    if (n <= chunk_size) {
      preds <- predict(rf_model, newdata = test_set)
    } else {
      preds <- numeric(n)
      starts <- seq(1L, n, by = chunk_size)
      for (i in seq_along(starts)) {
        idx <- starts[i]:min(starts[i] + chunk_size - 1L, n)
        preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
      }
    }

    # Fast by-reference update — no copy of the entire column
    pred_db[.(yr), consolidated := preds]

  } else {
    cat(paste0("  Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each year
  rm(test_set)
  if (exists("preds")) rm(preds)
  gc()
}

# ── Write output efficiently ─────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Optimization | Expected Impact |
|---|---|
| Remove unused libraries and `st_read` | Frees potentially **gigabytes** of RAM |
| Fix `filter(year == year)` masking bug | Reduces per-iteration rows from **N** to **N/T** (T = number of years), cutting CPU and memory proportionally |
| `data.table` keyed subsetting and `:=` assignment | Eliminates repeated full-scan indexing and copy-on-modify overhead |
| Chunked `predict()` | Caps peak memory during prediction at ~50 K rows regardless of year size |
| `fwrite` instead of `write.csv` | Typically **5–10×** faster for large outputs |

These changes preserve the trained Random Forest models exactly as-is and require no retraining.