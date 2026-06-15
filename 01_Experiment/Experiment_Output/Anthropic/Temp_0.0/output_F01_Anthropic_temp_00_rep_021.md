 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Loading a large `.RData` file entirely into memory:** The file likely contains the training data (`joined_data`), the prediction frame (`pred_db`), and the model list (`rf_models_per_year`). The training data and any other residual objects occupy memory needlessly during prediction.

4. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr::filter` to compare the column to itself (always `TRUE`), meaning **every row** is selected every iteration instead of just the target year. This makes each prediction pass operate on the full dataset — massively inflating computation time and producing incorrect results.

5. **Predicting on `joined_data` but assigning to `pred_db`:** If these two data frames have different row counts or orderings, the assignment `pred_db$consolidated[pred_db$year == year]` may silently produce wrong results or errors.

6. **`data.frame` column assignment in a loop:** Repeatedly assigning into a large data frame column inside a loop is slow due to R's copy-on-modify semantics.

7. **Writing a massive CSV:** For hundreds of thousands of rows with many columns, `write.csv` is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Load only the required libraries | Reduce memory footprint |
| 2 | Remove the unused `st_read` call | Avoid loading a large spatial object |
| 3 | After loading `.RData`, remove all objects except `rf_models_per_year` and `pred_db` (or `joined_data`, whichever is the correct prediction frame) | Free memory |
| 4 | Fix the variable masking bug by using `.env$year` or renaming the loop variable | Correctness — also fixes the performance problem of predicting on the full dataset each iteration |
| 5 | Use `data.table` for the prediction target and assign by reference | Avoid repeated data frame copies |
| 6 | Use `data.table::fwrite` for output | Much faster I/O |
| 7 | Call `gc()` after large memory releases | Prompt garbage collection |
| 8 | Preserve trained RF models exactly as-is (no retraining) | Per requirement |

---

## Optimized R Code

```r
# ── 1. Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── 2. Load saved models and prediction data ─────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# After loading, the workspace should contain at least:
#   rf_models_per_year  – named list of trained RF models
#   joined_data         – data frame used for prediction features
#   pred_db             – data frame where predictions are stored
#
# If pred_db and joined_data are the same object, adjust accordingly.

# ── 3. Drop every object we do not need to free RAM ─────────────────────────
keep_objects <- c("rf_models_per_year", "joined_data", "pred_db")
rm(list = setdiff(ls(), keep_objects))
gc()

# ── 4. Convert to data.table for fast by-reference operations ────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── 5. Get the years for which we have models ────────────────────────────────
model_years <- names(rf_models_per_year)
unique_years <- unique(pred_db$year)

# ── 6. Prediction loop — fixed variable-masking bug ─────────────────────────
for (yr in unique_years) {

  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% model_years) {

    # Correct subset: compare column to the external loop variable
    idx_joined <- which(joined_data$year == yr)
    idx_pred   <- which(pred_db$year == yr)

    # Safety check
    if (length(idx_joined) != length(idx_pred)) {
      warning(sprintf(
        "Row count mismatch for year %s: joined_data has %d rows, pred_db has %d rows.",
        yr_char, length(idx_joined), length(idx_pred)
      ))
    }

    test_subset <- joined_data[idx_joined, ]

    preds <- predict(rf_models_per_year[[yr_char]], newdata = test_subset)

    # Assign by reference (no copy of the whole table)
    set(pred_db, i = idx_pred, j = "consolidated", value = preds)

  } else {
    cat(sprintf("Warning: No model found for year %s\n", yr_char))
  }
}

# ── 7. Write output efficiently ──────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Improvements

- **Correctness fix:** The `filter(year == year)` bug was the single largest source of both incorrect results and wasted computation. Each iteration was predicting on the *entire* dataset instead of one year's slice.
- **Memory reduction:** Removing unused libraries, the unused shapefile read, and residual objects from the `.RData` file can free gigabytes of RAM.
- **Speed improvement:** `data.table::set()` assigns by reference without triggering a full-table copy, and `fwrite` is typically 5–10× faster than `write.csv` for large files.
- **Models are untouched:** No retraining occurs; the pre-trained `rf_models_per_year` list is used exactly as loaded.