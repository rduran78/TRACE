 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (due to the loop variable sharing the same name), meaning `test_set` always equals the entire `joined_data` dataframe — every year predicts on *all* rows, massively inflating computation time and producing wrong results.

4. **Monolithic `predict()` call on huge data:** Calling `predict()` on hundreds of thousands of rows at once with a large Random Forest model can spike memory usage. Batching is safer.

5. **`pred_db` may be a large dataframe held entirely in memory** while also writing a potentially huge CSV at the end.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read()` call entirely |
| Variable shadowing (`year == year`) | Rename the loop variable (e.g., `yr`) so `dplyr::filter` works correctly |
| Large single `predict()` calls | Predict in batches (chunks) to control peak memory |
| Memory pressure | Use `gc()` between years; use `data.table` for subsetting instead of `dplyr` |
| Large CSV output | Write results incrementally or use `fwrite()` from `data.table` for speed |

The trained Random Forest models (`rf_models_per_year`) are **preserved and never retrained**.

---

## Optimized R Code

```r
# ── Load only necessary libraries ──────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models ────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert working dataframes to data.table for speed ─────────
# 'joined_data' and 'pred_db' are expected to come from the .RData file.
setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# ── Batch-predict helper (controls peak memory) ────────────────
predict_in_batches <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Main prediction loop ──────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correct subsetting — no variable-shadowing bug
    idx_test <- which(joined_data$year == yr)
    idx_pred <- which(pred_db$year == yr)

    test_set <- joined_data[idx_test, ]

    rf_model <- rf_models_per_year[[yr_char]]
    pred_db[idx_pred, consolidated := predict_in_batches(rf_model, test_set)]

    # Free temporary objects and reclaim memory
    rm(test_set, rf_model)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output efficiently ──────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

- **Removed** all unused libraries and the unused `st_read()` call → large memory savings.
- **Fixed the shadowing bug** by renaming the loop variable to `yr`, so `joined_data$year == yr` filters correctly.
- **Switched to `data.table`** for fast subsetting via `which()` and in-place assignment via `:=`.
- **Added batch prediction** (`predict_in_batches`) to cap peak memory during `predict()`.
- **Added `rm()` + `gc()`** after each year to release memory promptly.
- **Replaced `write.csv` with `fwrite`** for significantly faster output.
- **Models are never retrained**; only `predict()` is called on existing `rf_models_per_year`.