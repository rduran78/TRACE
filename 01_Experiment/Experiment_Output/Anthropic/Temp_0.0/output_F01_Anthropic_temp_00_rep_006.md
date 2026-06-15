 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` always equals the entire `joined_data` dataframe. This means every iteration predicts on *all* rows, not just the target year, wasting compute and producing incorrect results.

4. **Whole-dataframe column assignment in a loop:** Writing into `pred_db$consolidated[pred_db$year == year]` inside a loop over years is slow for large data frames due to repeated subsetting and copy-on-modify behavior.

5. **`predict.randomForest` on massive data:** Calling `predict()` on hundreds of thousands of rows at once with a large Random Forest can spike memory because the function internally allocates matrices proportional to `n_rows × n_trees`.

6. **Using `data.frame` instead of `data.table`:** Base R data frames and `dplyr` operations are slower than `data.table` for large row-level operations.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consuming memory | Remove all unused `library()` calls |
| Unused spatial data (`prep_data`) | Remove the `st_read()` call entirely |
| Variable shadowing bug in `filter()` | Rename the loop variable (e.g., `yr`) so it doesn't collide with the column name |
| Slow row-level assignment in a loop | Use `data.table` keyed joins or pre-allocate a results list and bind once |
| Memory spike from predicting all rows at once | Predict in chunks (batches) within each year |
| Large `.RData` file loads everything | If possible, only load the needed objects with targeted extraction |

The trained Random Forest models (`rf_models_per_year`) are **preserved exactly as-is** — no retraining occurs.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and prediction data ──────────────────────
# This loads rf_models_per_year, joined_data, and pred_db.
# If the .RData file is very large and contains many unneeded objects,
# consider saving only the required objects to a smaller file in advance.
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed ─────────────────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Pre-allocate the output column ───────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Batch-prediction settings ────────────────────────────────────────
BATCH_SIZE <- 50000L   # Adjust based on available RAM

# ── Predict year by year ─────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── Subset only the rows for this year (bug-fixed filter) ──────────
  test_set <- joined_data[year == yr]

  n <- nrow(test_set)
  preds <- numeric(n)

  # ── Predict in chunks to limit peak memory ─────────────────────────
  starts <- seq(1L, n, by = BATCH_SIZE)

  for (s in starts) {
    e <- min(s + BATCH_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # ── Write predictions back (keyed assignment, no copy) ─────────────
  pred_db[year == yr, consolidated := preds]

  # ── Free memory before next year ───────────────────────────────────
  rm(test_set, preds)
  gc()
}

# ── Write output ─────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")    # data.table::fwrite is much faster than write.csv

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Removed 12 unused library calls and the unused `st_read()` | Saves hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` to fix the shadowing bug | Correctness — each year now filters properly |
| 3 | Converted to `data.table` with `:=` assignment | Avoids copy-on-modify; faster subsetting |
| 4 | Added chunked (batched) prediction | Caps peak memory per `predict()` call |
| 5 | Added `rm()` + `gc()` after each year | Frees intermediate objects promptly |
| 6 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster CSV writing |
| 7 | Models (`rf_models_per_year`) are untouched | No retraining; models are used as-is |