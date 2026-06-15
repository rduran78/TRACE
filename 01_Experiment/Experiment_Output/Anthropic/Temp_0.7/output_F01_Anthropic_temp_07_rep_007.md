 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but is never used in the prediction workflow. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), returning the **entire dataset** every iteration rather than the subset for that year. This massively inflates computation per iteration.

4. **Predicting on the full dataset repeatedly:** Because of bug #3, `predict()` is called on the entire `joined_data` for every year — multiplying runtime by the number of years.

5. **`predict.randomForest` on huge data:** Even with the bug fixed, calling `predict()` on hundreds of thousands of rows with many predictors in a single call can spike memory. Batching can help.

6. **Using a `data.frame` for row-level assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` repeatedly on a large data.frame is slow. Using `data.table` is far more efficient.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the unused `st_read()` call | Free memory immediately |
| Fix the variable masking bug (`year == year`) | Predict only on the correct yearly subset |
| Convert `pred_db` and `joined_data` to `data.table` | Faster subsetting and assignment by reference |
| Batch the `predict()` call if subsets are still very large | Cap peak memory usage |
| Trigger garbage collection after each year | Reclaim memory between iterations |
| No model retraining | Models are preserved as-is |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)

# ── Load the pre-trained models and associated data ──────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast operations ────────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Initialize the output column ─────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Define a batched predict function to cap memory usage ────────────
predict_batched <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  if (n <= batch_size) {
    return(predict(model, newdata = newdata))
  }
  # Pre-allocate output
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Prediction loop (with masking bug fixed) ─────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Correct subset: use the loop variable 'yr', not the column name
    test_set <- joined_data[year == yr]

    preds <- predict_batched(rf_model, newdata = test_set, batch_size = 50000L)

    # Fast assignment by reference
    pred_db[year == yr, consolidated := preds]

    # Free memory
    rm(test_set, preds, rf_model)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output ─────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Effect |
|---|---|---|
| 1 | Removed 11 unused library calls and the unused `st_read()` | Saves hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` to fix `filter(year == year)` masking bug | Predicts on the correct yearly subset instead of the entire dataset |
| 3 | Replaced `dplyr` + `data.frame` with `data.table` | Subsetting and `:=` assignment by reference are orders of magnitude faster |
| 4 | Added `predict_batched()` with a configurable `batch_size` | Caps peak memory during `predict()` on large subsets |
| 5 | Added `rm()` + `gc()` after each year | Frees memory between iterations |
| 6 | Replaced `write.csv()` with `fwrite()` | Much faster CSV writing for large tables |
| 7 | **No model retraining** | All original `rf_models_per_year` objects are used as-is |