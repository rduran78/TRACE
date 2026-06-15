 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction loop. This wastes significant memory.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning **every row** is selected every iteration — not just the target year. This is both a correctness bug and a massive performance problem because the model predicts on the entire dataset each iteration.

4. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` triggers repeated memory copies of the full data frame (copy-on-modify semantics in R).

5. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive, especially with large forests. No chunking or garbage collection is performed.

6. **`write.csv` on a very large data frame** is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries and objects consume memory | Remove all unused `library()` calls; do not load `prep_data` |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) and use `.env$yr` or base-R subsetting |
| Repeated full-dataframe copies on assignment | Convert `pred_db` to a `data.table` and assign by reference with `:=` |
| Large single-batch predictions | Optionally chunk predictions within each year and call `gc()` between years |
| Slow CSV writing | Use `data.table::fwrite` |
| All models held in memory simultaneously | Optionally load/unload models one at a time if memory is critical |

---

## Optimized R Code

```r
# ── Only load what is actually needed ──
library(data.table)
library(randomForest)

# ── Load the trained models and associated data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for fast by-reference operations ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {

    # Correctly subset to the current year (avoids the masking bug)
    test_set <- joined_data[year == yr]

    rf_model <- rf_models_per_year[[yr_char]]

    # ── Optional chunking for very large year-slices ──
    n        <- nrow(test_set)
    chunk_sz <- 50000L
    preds    <- numeric(n)

    if (n > chunk_sz) {
      starts <- seq(1L, n, by = chunk_sz)
      for (s in starts) {
        e <- min(s + chunk_sz - 1L, n)
        preds[s:e] <- predict(rf_model, newdata = test_set[s:e])
      }
    } else {
      preds <- predict(rf_model, newdata = test_set)
    }

    # Assign by reference — no copy of the whole table
    pred_db[year == yr, consolidated := preds]

    # Free memory before next iteration
    rm(test_set, preds, rf_model)
    gc()

  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Why |
|---|---|---|
| 1 | Removed 12 unused libraries | Reduces memory footprint and load time |
| 2 | Removed unused `st_read` call | Avoids loading a large spatial file into RAM |
| 3 | Renamed loop variable to `yr` and used `data.table` syntax `joined_data[year == yr]` | **Fixes the variable-masking bug** that caused incorrect and bloated predictions |
| 4 | Converted `pred_db` and `joined_data` to `data.table`; used `:=` for assignment | Eliminates repeated full-table copies (O(1) update instead of O(n)) |
| 5 | Added chunked prediction with configurable `chunk_sz` | Caps peak memory during `predict()` |
| 6 | Added `rm()` + `gc()` after each year | Frees memory between iterations |
| 7 | Replaced `write.csv` with `fwrite` | Typically 10–50× faster for large files |

No models are retrained; all original `rf_models_per_year` objects are used as-is.