 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains the **entire** dataset every iteration — not just one year's worth. This multiplies computation time by the number of unique years.

4. **Using a full `data.frame` (`joined_data`) instead of `data.table`:** For hundreds of thousands of rows, subsetting a `data.frame` or `tibble` repeatedly is slower than `data.table` keyed subsetting.

5. **Writing the full `pred_db` (potentially with many columns) to CSV:** If `pred_db` is large, this is slow and disk-intensive. Only necessary columns should be written.

6. **`predict.randomForest` on massive data in one call:** For very large test sets, `predict()` on a Random Forest can spike memory because it internally allocates matrices across all trees. Batching predictions can keep memory bounded.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the `st_read()` call | Free significant memory |
| Fix the variable masking bug (`year == year`) | Avoid predicting on the full dataset every iteration |
| Convert `joined_data` to `data.table` and key by year | Faster subsetting |
| Batch predictions within each year | Bound peak memory usage |
| Write only necessary output columns | Reduce I/O time and disk usage |
| Use `fwrite()` instead of `write.csv()` | Much faster CSV writing |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data was built as tibble

# ── Load pre-trained models and prediction database ──────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast keyed operations ──────────────────
setDT(joined_data)
setDT(pred_db)

# Key by year for fast subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Initialise the output column
pred_db[, consolidated := NA_real_]

# ── Batch-prediction helper (bounds memory per chunk) ────────────────
predict_batched <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  if (n <= batch_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Main prediction loop (with corrected filter) ────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correct subsetting — no variable masking
    test_set <- joined_data[.(yr)]          # data.table keyed lookup
    rf_model <- rf_models_per_year[[yr_char]]

    preds <- predict_batched(rf_model, test_set, batch_size = 50000L)

    pred_db[.(yr), consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each year
  gc()
}

# ── Write output efficiently (only needed columns) ───────────────────
# Adjust the id column name to whatever uniquely identifies each cell-year
output_cols <- intersect(names(pred_db), c("cell_id", "year", "consolidated"))
fwrite(pred_db[, ..output_cols], "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| Original Problem | Fix Applied |
|---|---|
| ~10 unused libraries consuming RAM | Removed all unused `library()` calls and the `st_read()` call |
| `filter(year == year)` selects **all** rows every iteration | Replaced with `data.table` keyed lookup `joined_data[.(yr)]` using an unambiguous loop variable `yr` |
| Single massive `predict()` call can spike memory | Wrapped in `predict_batched()` with a configurable `batch_size` |
| Slow `write.csv()` on large output | Replaced with `data.table::fwrite()` on only the essential columns |
| No memory cleanup between years | Added `gc()` after each year's prediction |

These changes preserve all trained Random Forest models exactly as-is (no retraining) and should make the workflow feasible on a standard personal computer.