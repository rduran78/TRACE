 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geographic cells.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning **every row** is selected every iteration rather than just the target year. This is both a correctness bug and a performance disaster—every year's prediction runs on the entire dataset.

4. **Data frame column assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` row-by-row within a `for` loop on a large data frame is slow due to repeated copying.

5. **Using `predict.randomForest` on massive data frames:** With hundreds of thousands of rows and many predictors, a single `predict()` call can spike memory. No batching is used.

6. **Writing a massive CSV with `write.csv`:** `write.csv` is slow for large data. `data.table::fwrite` is dramatically faster.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused library calls | Reduce memory footprint |
| Remove unused `prep_data` read | Free potentially gigabytes of RAM |
| Fix the `year == year` masking bug | Correctness + avoids predicting on full dataset each iteration |
| Convert to `data.table` operations | Faster subsetting and assignment by reference |
| Batch predictions within each year | Cap peak memory usage |
| Use `fwrite` instead of `write.csv` | Much faster I/O for large files |
| Explicitly call `gc()` between years | Release memory between iterations |

**The trained Random Forest models are fully preserved and never retrained.**

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)

# ── Load the saved models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load prep_data — it is unused and wastes memory.
# prep_data <- st_read(...)  # REMOVED

# ── Convert to data.table for fast by-reference operations ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Define a batch-predict helper to limit peak memory ──
batch_predict <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Predict year by year ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correctly subset using the loop variable (not the column name)
    test_set <- joined_data[year == yr]

    rf_model <- rf_models_per_year[[yr_char]]

    # Batch predict to control memory usage
    preds <- batch_predict(rf_model, test_set, batch_size = 50000L)

    # Assign by reference — no copy of the entire data.table
    pred_db[year == yr, consolidated := preds]

    # Free temporary objects and reclaim memory
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output with fast I/O ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| Original Problem | Fix Applied |
|---|---|
| ~15 unused libraries loaded | Reduced to 2 (`randomForest`, `data.table`) |
| Large shapefile loaded but unused | Removed entirely |
| `filter(year == year)` selects all rows (bug) | Renamed loop variable to `yr`; uses `data.table` syntax `joined_data[year == yr]` |
| Slow `data.frame` subset-assignment in loop | `data.table` assignment by reference (`:=`) |
| Unbounded memory on large `predict()` calls | `batch_predict()` processes in chunks of 50,000 rows |
| Slow `write.csv` | Replaced with `data.table::fwrite` |
| No memory cleanup between iterations | `rm()` + `gc()` after each year |

These changes should make the prediction pipeline feasible on a standard personal computer without retraining any models.