 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not work as intended. The loop variable `year` shadows the column name `year`, so the filter matches *all* rows every iteration instead of just the target year. This means the `predict()` call processes the entire dataset every iteration, massively inflating computation time.

4. **Using `data.frame` operations on large data:** Subsetting and assigning with base R on a large `data.frame` (especially one that may carry `sf` geometry or `tibble` overhead) is slow.

5. **Writing the full object with `write.csv`:** For hundreds of thousands of rows with many columns, `write.csv` is slow compared to `data.table::fwrite`.

6. **No memory cleanup:** The full `joined_data` and all models sit in memory simultaneously.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and objects | Remove `st_read()` call and unneeded `library()` calls to free memory. |
| Variable masking bug | Rename the loop variable (e.g., `yr`) so `dplyr::filter` correctly subsets by year. |
| Predicting on full data each iteration | Fix the filter so only the relevant year's rows are passed to `predict()`. |
| Slow subsetting/assignment | Convert `pred_db` and `joined_data` to `data.table` for fast keyed operations. |
| Slow CSV writing | Use `data.table::fwrite()`. |
| Memory pressure | Use `gc()` after large operations; remove objects no longer needed. |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(data.table)
library(randomForest)   # needed for predict() on randomForest objects

# ── Load the saved workspace (contains rf_models_per_year,
#    pred_db, joined_data, etc.) ──────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Remove any large objects from the .RData that are not needed ─────
# (Uncomment and adapt if other large objects exist in the workspace)
# rm(object_not_needed); gc()

# ── Convert to data.table for fast keyed operations ──────────────────
setDT(pred_db)
setDT(joined_data)

# Create keys for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Initialise the output column
pred_db[, consolidated := NA_real_]

# ── Prediction loop (fixed variable-masking bug) ─────────────────────
years_to_predict <- unique(pred_db$year)

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Fast keyed subset — only the rows for this year
    test_set <- joined_data[.(yr)]

    rf_model <- rf_models_per_year[[yr_char]]
    preds    <- predict(rf_model, newdata = test_set)

    # Fast keyed assignment
    pred_db[.(yr), consolidated := preds]

    # Free memory from this iteration
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write results efficiently ────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Aspect | Before | After |
|---|---|---|
| **Memory at load** | ~2× larger (unused spatial shapefile loaded) | Shapefile never loaded |
| **Rows sent to `predict()` per iteration** | Entire dataset (filter bug) | Only the target year's rows |
| **Subsetting speed** | Base R / dplyr on data.frame | `data.table` keyed lookup (O(log n)) |
| **Assignment speed** | Base R bracket assignment | `data.table` keyed `:=` (in-place) |
| **CSV write speed** | `write.csv` (single-threaded, slow) | `fwrite` (multi-threaded, fast) |
| **Idle memory** | Accumulated across loop | Freed each iteration with `rm()` + `gc()` |

The models (`rf_models_per_year`) are **never retrained**; only `predict()` is called. The critical correctness fix (renaming the loop variable from `year` to `yr`) alone may reduce total runtime by a factor roughly equal to the number of distinct years, since the original code was inadvertently predicting on the full dataset in every single iteration.