 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in this script:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (since the loop variable and the column share the same name), returning *all* rows every iteration instead of the intended subset. This is a critical correctness bug that also amplifies the performance problem.

2. **Unnecessary Loading of Heavy Spatial Data:** `prep_data` is loaded via `st_read` but never used in the prediction loop. Reading a large shapefile into memory wastes RAM.

3. **Massive In-Memory Data Frames:** With hundreds of thousands of cells × many years × many predictor columns, both `joined_data` and `pred_db` may consume multiple gigabytes. Subsetting a `data.frame`/`tibble` with `dplyr::filter` inside a loop creates a full copy each iteration.

4. **Row-Level Assignment in a Loop:** `pred_db$consolidated[pred_db$year == year]` scans the entire data frame for matching rows on every iteration.

5. **`write.csv` on a Huge Data Frame:** This is slow; `data.table::fwrite` is dramatically faster.

6. **Library Bloat:** Many loaded packages (blockCV, zoo, LongituRF, sf, spdep, plm, fixest, scales, stringi, terra) are unused during prediction, consuming memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename loop variable to `yr` |
| Unused `prep_data` load | Remove the `st_read` call |
| Unnecessary libraries | Load only what is needed for prediction |
| Slow subsetting & assignment | Convert to `data.table`, key by `year`, subset by reference |
| Full data copy per iteration | Use `data.table` in-place update with `:=` |
| `predict()` on huge sets | Optionally chunk predictions within each year |
| Slow CSV write | Use `fwrite` |

---

## Optimized Working Code

```r
# ── Load only the libraries needed for prediction ──────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load the saved models (and the associated data objects) ────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, joined_data, pred_db

# ── Convert both data frames to data.tables for speed ─────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Pre-allocate the output column ────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Key both tables on year for fast subsetting ───────────────────────────
setkey(pred_db,     year)
setkey(joined_data, year)

# ── Identify which years have a trained model ─────────────────────────────
model_years <- names(rf_models_per_year)

# ── Prediction loop (note: loop variable is 'yr' to avoid shadowing) ──────
for (yr in unique(pred_db$year)) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% model_years) {
    rf_model  <- rf_models_per_year[[yr_char]]

    # Fast keyed subset — no copy of the full table
    test_rows <- joined_data[.(yr)]

    # ── Optional: chunk predictions to limit peak memory ──────────────
    chunk_size <- 50000L
    n          <- nrow(test_rows)

    if (n <= chunk_size) {
      preds <- predict(rf_model, newdata = test_rows)
    } else {
      preds <- numeric(n)
      starts <- seq(1L, n, by = chunk_size)
      for (i in seq_along(starts)) {
        idx_start <- starts[i]
        idx_end   <- min(idx_start + chunk_size - 1L, n)
        preds[idx_start:idx_end] <-
          predict(rf_model, newdata = test_rows[idx_start:idx_end])
      }
    }

    # In-place update — no whole-table copy
    pred_db[.(yr), consolidated := preds]

  } else {
    cat(sprintf("  Warning: No model found for year %s\n", yr))
  }

  # Free temporary objects after each year
  rm(test_rows, preds)
  gc()
}

# ── Fast write ────────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

1. **Bug fix:** Renamed the loop variable to `yr` so that `joined_data[.(yr)]` correctly subsets rows for that single year.
2. **Removed unused `st_read`** call and all unused `library()` calls → frees several GB of RAM.
3. **Converted to `data.table`** with keyed subsetting and `:=` in-place assignment → eliminates repeated full-table scans and copies.
4. **Added chunked `predict()`** (configurable `chunk_size`) so that if a single year has hundreds of thousands of rows, predictions are generated in manageable batches, keeping peak memory bounded.
5. **Replaced `write.csv` with `fwrite`** → typically 10–50× faster on large data.
6. **Explicit `gc()` per iteration** to return memory to the OS between years.

These changes preserve all trained Random Forest models exactly as-is and require no retraining.