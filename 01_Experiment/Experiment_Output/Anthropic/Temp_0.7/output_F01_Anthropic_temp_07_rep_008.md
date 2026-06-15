 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geometries.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly because the column name `year` and the loop variable `year` are the same. This causes `dplyr` to compare the column to itself, returning **all rows** every iteration — massively inflating computation and producing incorrect results.

4. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` performs repeated subsetting on a potentially very large data frame, which is slow.

5. **`predict.randomForest` on massive data:** With hundreds of thousands of rows and many trees, `predict()` can be memory-intensive. This is unavoidable but can be helped by reducing ambient memory pressure.

6. **Writing a huge CSV:** `write.csv` on a large data frame is slow; `data.table::fwrite` is substantially faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries & objects | Remove them to free memory |
| Unused shapefile load | Remove `st_read()` call entirely |
| Variable masking bug | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Slow subsetting in loop | Pre-split data with `split()`, collect results in a list, then `rbindlist()` |
| Memory pressure during `predict()` | Use `gc()` between years; optionally predict in row-chunks |
| Slow CSV write | Use `data.table::fwrite()` |

---

## Optimized R Code

```r
# ── Load only what is needed ──
library(randomForest)   # for predict()
library(data.table)     # for fast I/O and binding
library(dplyr)          # for minimal data manipulation

# ── Load pre-trained models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year (named list), joined_data, pred_db

# ── Drop any large objects that are not needed ──
# (If the .RData file contains other objects, remove them)
# rm(any_unneeded_object); gc()

# ── Convert to data.table for speed ──
setDT(pred_db)
setDT(joined_data)

# ── Initialize output column ──
pred_db[, consolidated := NA_real_]

# ── Pre-split joined_data by year (done once, avoids repeated filtering) ──
joined_splits <- split(joined_data, by = "year")

# ── Available model years ──
model_years <- names(rf_models_per_year)

# ── Predict year by year ──
for (yr in unique(pred_db$year)) {
  yr_char <- as.character(yr)
  cat(paste0("Predicting for year ", yr, " ... "))

  if (yr_char %in% model_years) {

    # Retrieve the pre-split test data for this year
    test_set <- joined_splits[[yr_char]]

    if (is.null(test_set) || nrow(test_set) == 0L) {
      cat("no test data found, skipping.\n")
      next
    }

    rf_model <- rf_models_per_year[[yr_char]]

    # ── Optional: chunk-wise prediction to limit peak memory ──
    n        <- nrow(test_set)
    chunk_sz <- 50000L
    preds    <- numeric(n)

    for (start in seq(1L, n, by = chunk_sz)) {
      end <- min(start + chunk_sz - 1L, n)
      preds[start:end] <- predict(rf_model, newdata = test_set[start:end, ])
    }

    # Write predictions back (keyed assignment in data.table is fast)
    pred_db[year == yr, consolidated := preds]

    # Free memory before next iteration
    rm(rf_model, test_set, preds)
    gc()

    cat("done.\n")
  } else {
    cat(paste0("WARNING – no model found, skipping.\n"))
  }
}

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Removed** all unused libraries and the unused `st_read()` call → **large memory savings**.
2. **Fixed the filtering bug** by renaming the loop variable to `yr`, so `year == yr` filters correctly.
3. **Pre-split** `joined_data` once by year with `split()` → avoids repeated full-table scans.
4. **Chunk-wise `predict()`** (50 000 rows at a time) → caps peak memory during prediction.
5. **`data.table` throughout** → faster subsetting, assignment, and writing.
6. **Explicit `rm()` + `gc()`** after each year → reclaims memory between iterations.
7. **No models are retrained** — all existing `rf_models_per_year` objects are used as-is.