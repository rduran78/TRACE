 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data (shapefiles) with hundreds of thousands of features can consume gigabytes of RAM.

3. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on the **full dataset**, massively inflating computation time and producing incorrect results.

4. **Predicting all at once per year:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows in a single call with a large Random Forest model can spike memory usage because `randomForest::predict` builds a full matrix of tree-level predictions internally.

5. **Using a data.frame for repeated row-assignment:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a large data.frame inside a loop is slow due to repeated logical indexing and potential copy-on-modify behavior.

6. **Writing a massive CSV:** `write.csv` on a very large data.frame is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Remove unused libraries and the unused `st_read()` call | Free memory immediately |
| 2 | Fix the variable shadowing bug in `filter()` | Correctness — without this, every year predicts on the full dataset |
| 3 | Convert `pred_db` and `joined_data` to `data.table` | Faster subsetting, in-place assignment via `:=`, no copy-on-modify |
| 4 | Predict in chunks (batched prediction) | Caps peak memory during `predict()` calls |
| 5 | Explicitly garbage-collect after each year | Reclaim memory between iterations |
| 6 | Use `data.table::fwrite` for output | Much faster than `write.csv` |
| 7 | Do **not** retrain any model | Preserved as required |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data was built with dplyr

# ── Load the pre-trained models and prediction database ──────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: pred_db, joined_data, rf_models_per_year

# ── Convert to data.table for speed and memory efficiency ────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Batched predict function to cap memory usage ─────────────────────
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

# ── Main prediction loop (with corrected filter) ────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correct subsetting — use the loop variable, not the column name
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

# ── Write output efficiently ─────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

- **Memory savings:** Removed the unused `st_read()` call and unnecessary libraries, potentially freeing gigabytes of RAM.
- **Correctness fix:** Renamed the loop variable to `yr` so that `joined_data[year == yr]` correctly subsets a single year instead of returning the entire dataset.
- **Speed & memory:** Converted to `data.table` for in-place `:=` assignment (no copy-on-modify), chunked `predict()` to cap peak memory, and used `fwrite` for fast CSV output.
- **Models untouched:** No retraining occurs; the pre-trained `rf_models_per_year` are used as-is.