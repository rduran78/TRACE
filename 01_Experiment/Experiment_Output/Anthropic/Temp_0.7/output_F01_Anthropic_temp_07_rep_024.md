 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (not to the loop variable), returning *all rows* every iteration. This means every year's prediction runs on the entire dataset — massively inflating compute time and producing incorrect results.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used, consuming significant memory for a large shapefile.

3. **Full Dataset in Memory:** `joined_data` and `pred_db` are likely large `data.frame` or `sf` objects. Holding them entirely in memory alongside the Random Forest models (which can be very large) may exceed RAM.

4. **Inefficient Row Assignment:** `pred_db$consolidated[pred_db$year == year]` performs a full-column logical scan on every iteration.

5. **Unused Libraries:** Many loaded libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, etc.) are never used in the prediction loop, adding overhead.

6. **CSV Output of Huge Data:** `write.csv()` on hundreds of thousands of rows is slow; `data.table::fwrite()` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so `filter(year == yr)` works correctly |
| Unused shapefile load | Remove `st_read()` call |
| Unused libraries | Load only what is needed |
| Large objects in memory | Convert to `data.table`; drop unneeded columns; process year-by-year and free memory |
| Slow row indexing | Use `data.table` keyed joins / set-by-reference |
| Slow CSV write | Use `fwrite()` |
| Optional: parallelism | Not needed once the bug is fixed (each year's subset is now much smaller) |

The trained Random Forest models are **preserved untouched** — no retraining occurs.

---

## Optimized R Code

```r
# ── Load only required libraries ──────────────────────────────────────────────
library(data.table)
library(randomForest)   # needed for predict() on rf objects
library(dplyr)          # only if joined_data is a tibble; can be dropped if converted

# ── Load pre-trained models and prediction database ───────────────────────────
# (Do NOT load the unused shapefile)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: pred_db, joined_data, rf_models_per_year

# ── Convert to data.table for speed and memory efficiency ─────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Pre-allocate the output column ────────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Identify the predictor columns the models expect ──────────────────────────
# (Grab from any one model so we only keep needed columns in the subset)
sample_model <- rf_models_per_year[[1]]
model_vars   <- rownames(importance(sample_model))

# ── Key joined_data by year for fast subsetting ───────────────────────────────
setkey(joined_data, year)
setkey(pred_db, year)

# ── Predict year-by-year ──────────────────────────────────────────────────────
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {
  yr_char <- as.character(yr)
  cat(paste0("Predicting for year ", yr, " ... "))

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("WARNING: no model found — skipping.\n")
    next
  }

  # Fast keyed subset; keep only columns the model needs (saves memory)
  test_set <- joined_data[.(yr), .SD, .SDcols = intersect(model_vars, names(joined_data))]

  rf_model <- rf_models_per_year[[yr_char]]
  preds    <- predict(rf_model, newdata = test_set)

  # Set by reference — no copy of the whole column
  pred_db[.(yr), consolidated := preds]

  cat(paste0(length(preds), " cells predicted.\n"))

  # Free temporary objects to keep memory low
  rm(test_set, preds)
  gc()
}

# ── Write output efficiently ──────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Impact |
|---|--------|--------|
| 1 | **Fixed the `year == year` bug** by using loop variable `yr` | Correctness + massive speed gain (subset is now 1/N-th the size) |
| 2 | Removed unused `st_read()` | Saves potentially GBs of RAM |
| 3 | Removed 11 unused `library()` calls | Faster startup, less memory |
| 4 | Converted to `data.table` with keys | O(log n) subsetting; set-by-reference avoids copies |
| 5 | Subsetted only model-required columns (`model_vars`) | Reduces per-iteration memory |
| 6 | Added `rm()` + `gc()` inside loop | Frees memory each iteration |
| 7 | Replaced `write.csv()` with `fwrite()` | ~5–10× faster file writing |
| 8 | **No model retraining** | Models used as-is from the `.RData` file |

These changes should make the prediction loop feasible on a standard personal computer with moderate RAM (8–16 GB).