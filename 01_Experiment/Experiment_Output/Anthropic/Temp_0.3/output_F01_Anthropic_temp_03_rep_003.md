 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries can consume gigabytes of RAM.

3. **Variable masking bug:** `filter(year == year)` inside the `for (year in ...)` loop causes the column `year` to be compared to itself (always `TRUE`), meaning **every row** is selected every iteration instead of just the rows for that year. This massively inflates computation per iteration.

4. **Whole-dataframe writes per iteration:** `pred_db$consolidated[pred_db$year == year] <- ...` performs repeated subsetting on a potentially huge data frame, which is slow and memory-intensive.

5. **`predict()` on a large `randomForest` object with hundreds of thousands of rows** can spike memory, especially if the forest is large (many trees, many variables).

6. **Writing the full data frame to CSV at the end** with `write.csv` on a very large frame is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consuming memory | Remove all unused `library()` calls |
| Unused shapefile consuming RAM | Remove the `st_read()` call entirely |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) and filter with `.data$year == yr` or use `data.table` |
| Slow row-by-row data.frame subsetting | Convert to `data.table` and use keyed joins or split-apply |
| Memory spike from predicting all rows at once | Predict in chunks within each year if needed |
| Slow `write.csv` | Use `data.table::fwrite` |
| General overhead | Use `data.table` throughout; pre-split data by year |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)   # predict method for RF models
library(data.table)     # fast data manipulation and fwrite

# ── Load only the saved model workspace ──
# This should contain: rf_models_per_year, joined_data, pred_db
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed and memory efficiency ──
setDT(pred_db)
setDT(joined_data)

# ── Initialise the output column ──
pred_db[, consolidated := NA_real_]

# ── Pre-split the prediction features by year (avoids repeated filtering) ──
joined_splits <- split(joined_data, by = "year")

# ── Prediction loop with corrected variable scoping ──
available_years <- names(rf_models_per_year)

for (yr in unique(pred_db$year)) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% available_years) {
    # Retrieve the pre-split test data for this year
    test_set <- joined_splits[[yr_char]]

    if (is.null(test_set) || nrow(test_set) == 0L) {
      cat(paste0("  Warning: No test data for year ", yr, " – skipping.\n"))
      next
    }

    rf_model <- rf_models_per_year[[yr_char]]

    # ── Optional chunked prediction to limit peak memory ──
    n        <- nrow(test_set)
    chunk_sz <- 50000L
    preds    <- numeric(n)

    if (n > chunk_sz) {
      starts <- seq(1L, n, by = chunk_sz)
      for (s in starts) {
        e <- min(s + chunk_sz - 1L, n)
        preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
      }
    } else {
      preds <- predict(rf_model, newdata = test_set)
    }

    # Fast keyed update
    pred_db[year == yr, consolidated := preds]
  } else {
    cat(paste0("  Warning: No model found for year ", yr, "\n"))
  }
}

# ── Free memory occupied by objects no longer needed ──
rm(joined_data, joined_splits, rf_models_per_year)
gc()

# ── Write results with fast fwrite ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

- **Removed** 11 unused library imports and the unused shapefile read → significant RAM savings.
- **Fixed the critical bug:** renamed the loop iterator to `yr` so `filter`/subset correctly selects only the rows for that year, preventing the model from predicting on the entire dataset every iteration.
- **Pre-split** `joined_data` by year once, avoiding repeated full-table scans.
- **Chunked prediction** (default 50,000 rows per chunk) caps peak memory during `predict()`.
- **`data.table`** keyed assignment replaces slow base-R row indexing.
- **`fwrite`** replaces `write.csv` for much faster output.
- **Explicit `gc()`** after removing large objects reclaims memory.

These changes should make the workflow feasible on a standard personal computer without retraining any models.