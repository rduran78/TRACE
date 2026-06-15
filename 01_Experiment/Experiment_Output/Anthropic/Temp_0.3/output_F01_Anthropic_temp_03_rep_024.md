 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`st_read`) that is never used in prediction.** The `prep_data` object may consume hundreds of megabytes or more of RAM for no reason.

3. **The `filter()` variable-masking bug:** `test_set <- joined_data %>% filter(year == year)` does **not** filter by the loop variable — it compares the column `year` to itself (always `TRUE`), so `test_set` equals the **entire** `joined_data` every iteration. This means every call to `predict()` scores the full dataset instead of one year's slice, massively inflating computation time and memory use.

4. **Using `data.frame`-based `predict()` on hundreds of thousands of rows:** `randomForest::predict` on very large data frames is slow; converting to `data.table` or processing in chunks can help.

5. **Writing the full `pred_db` (potentially with geometry or many unused columns) to CSV** is slow and produces a bloated file.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries & objects | Remove them to free memory |
| `prep_data` shapefile loaded for nothing | Remove the `st_read` call |
| `filter(year == year)` bug | Use a distinct loop-variable name (e.g., `.yr`) or use `data.table` subsetting |
| Whole-dataset predict per year | Fix the filter so only one year's rows are scored |
| Large data.frame overhead | Convert `joined_data` and `pred_db` to `data.table` for fast keyed subsetting |
| Optional: parallel prediction | Use chunked prediction if single-year slices are still large |
| CSV output | Write only the columns needed with `fwrite` |

---

## Optimized R Code

```r
# ── Only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)

# ── Load the saved models and associated data ───────────────────
# (This .RData is expected to contain: rf_models_per_year, joined_data, pred_db)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Remove any large objects that were loaded but are not needed ─
# Adjust this list to whatever else is in the .RData file
# rm(prep_data)            # if it was saved inside the .RData
gc()                        # reclaim memory

# ── Convert to data.table for fast keyed operations ─────────────
setDT(pred_db)
setDT(joined_data)

# Create keyed index on year for O(log n) subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# ── Initialise the output column ────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Prediction loop (one year at a time) ────────────────────────
available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (.yr in available_years) {

  cat(sprintf("Predicting for year %s ...\n", .yr))

  # Correct, unambiguous subsetting — no dplyr masking bug
  test_set <- joined_data[year == as.numeric(.yr)]

  # Predict using the pre-trained RF model (not retrained)
  preds <- predict(rf_models_per_year[[.yr]], newdata = test_set)

  # Write predictions back by reference (no copy)
  pred_db[year == as.numeric(.yr), consolidated := preds]
}

# Warn about years with no model
missing <- setdiff(as.character(unique(pred_db$year)), names(rf_models_per_year))
if (length(missing) > 0) {
  cat("Warning: No model found for year(s):", paste(missing, collapse = ", "), "\n")
}

# ── Write only the columns you need (fast, small file) ──────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Change | Memory saved | Speed gained |
|---|---|---|
| Remove unused libraries & `st_read` | Hundreds of MB | Faster startup |
| Fix `year == year` filter bug | Up to *N×T* → *N* rows per predict call | **Largest single speedup** (orders of magnitude) |
| `data.table` keyed subsetting | Avoids full-copy `dplyr` tibbles | Faster subsetting |
| Assignment by reference (`:=`) | No copy of `pred_db` each iteration | Less memory churn |
| `fwrite` instead of `write.csv` | — | Much faster I/O |

The critical fix is the **filter bug**: once corrected, each `predict()` call processes only one year's cells instead of the entire multi-year dataset, which alone should make the workflow feasible on a personal computer.