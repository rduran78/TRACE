 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Loading a potentially massive `.RData` file entirely into memory:** The file `model_5_all_countries.RData` likely contains the training data (`joined_data`), the prediction frame (`pred_db`), and all year-specific Random Forest models (`rf_models_per_year`). All of these reside in memory simultaneously.

4. **Variable masking bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (due to name collision with the loop variable), returning **all rows** instead of the intended subset. This means every year's prediction runs on the entire dataset, which is both incorrect and extremely slow/memory-intensive.

5. **Using `data.frame` operations on large data:** `pred_db$consolidated[pred_db$year == year] <- ...` triggers repeated full-column scans and copy-on-modify behavior in base R.

6. **Writing a potentially huge CSV in one shot:** For hundreds of thousands of rows × many columns, `write.csv` is slow and memory-hungry.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so `dplyr::filter` correctly subsets |
| Entire dataset in memory at once | Process year-by-year, writing results incrementally, and free memory after each iteration |
| Slow base R assignment into `pred_db` | Convert to `data.table` and use keyed in-place update |
| Large CSV write | Use `data.table::fwrite` for fast, memory-efficient output |
| All RF models loaded simultaneously | Optionally, if `.RData` can be restructured, save/load one model at a time; otherwise, remove each model from the list after use to free memory |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(data.table)
library(randomForest)   # required for predict.randomForest
library(dplyr)          # used only for the filter step; can be replaced with data.table

# ── Do NOT load the unused shapefile ──────────────────────────────────────
# prep_data <- st_read(...)
# Removed: saves potentially gigabytes of RAM.

# ── Load saved objects ────────────────────────────────────────────────────
# This brings in: rf_models_per_year, pred_db, joined_data (at minimum)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast keyed operations ───────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# Create a key on year for fast subsetting
setkey(pred_db,    year)
setkey(joined_data, year)

# ── Year-by-year prediction loop ─────────────────────────────────────────
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {                       # 'yr' avoids name collision
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Subset using data.table key (very fast)
    test_set <- joined_data[.(yr)]                   # correct subset for this year

    # Predict
    preds <- predict(rf_model, newdata = test_set)

    # In-place update by reference (no copy)
    pred_db[.(yr), consolidated := preds]

    # Free the model from the list to release memory immediately
    rf_models_per_year[[yr_char]] <- NULL
    rm(rf_model, test_set, preds)
    gc()                                             # reclaim memory
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Fast, memory-efficient write ──────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed 11 unused libraries | Lower memory baseline |
| 2 | Removed unused `st_read` call | Avoids loading a large spatial object |
| 3 | Renamed loop variable to `yr` | **Fixes the bug** where `filter(year == year)` selected all rows |
| 4 | Converted to `data.table` with `setkey` | Subsetting and in-place update by reference; no copy-on-modify |
| 5 | Nullified each model after use (`rf_models_per_year[[yr_char]] <- NULL`) | Frees RAM progressively as predictions complete |
| 6 | Added `gc()` per iteration | Encourages R to return freed memory to the OS |
| 7 | Replaced `write.csv` with `fwrite` | Significantly faster and lower peak memory for large outputs |

No models are retrained; all existing Random Forest models are preserved and used as-is for prediction.