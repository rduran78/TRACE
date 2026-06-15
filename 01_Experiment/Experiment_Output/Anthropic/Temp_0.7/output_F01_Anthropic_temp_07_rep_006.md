 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Masking Bug:** `filter(year == year)` compares the column to itself (always TRUE), so every row is selected every iteration instead of just the rows for the target year. This means the full dataset is passed to `predict()` each loop, massively inflating memory and computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full Data Frame Copy per Iteration:** Subsetting a large `joined_data` data.frame inside a loop and writing predictions back row-by-row into `pred_db` via logical indexing is slow for hundreds of thousands of rows × many years.

4. **Using `data.frame` Instead of `data.table`:** Base R data.frame operations (logical subsetting, column assignment) are slower than `data.table` for large datasets.

5. **CSV Output for a Very Large File:** `write.csv()` is slow; `data.table::fwrite()` is significantly faster.

6. **All Libraries Loaded Upfront:** Many loaded libraries (e.g., `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`) are not needed for prediction, consuming memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable masking in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Slow subsetting/assignment | Convert to `data.table` and use keyed joins |
| Slow CSV write | Use `fwrite()` |
| Unnecessary libraries | Load only what is needed |
| Memory pressure | Use `gc()` between years; predict in place |

---

## Optimized Working R Code

```r
# ── Load only required libraries ──────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models ───────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert core data to data.table for speed ────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Pre-allocate the output column ────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Key both tables on year for fast subsetting ───────────────────
setkey(pred_db,    year)
setkey(joined_data, year)

# ── Determine which years have a trained model ────────────────────
model_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

# ── Predict year-by-year ──────────────────────────────────────────
for (yr in model_years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_num <- as.numeric(yr)

  # Fast keyed subset — avoids the variable-masking bug
  test_set <- joined_data[.(yr_num)]

  # Predict using the pre-trained Random Forest (no retraining)
  preds <- predict(rf_models_per_year[[yr]], newdata = test_set)

  # Fast in-place assignment via data.table
  pred_db[.(yr_num), consolidated := preds]

  # Free memory after each year
  rm(test_set, preds)
  gc()
}

# ── Report any years without a model ─────────────────────────────
missing <- setdiff(as.character(unique(pred_db$year)),
                   names(rf_models_per_year))
if (length(missing) > 0) {
  cat("Warning: No model found for year(s):", paste(missing, collapse = ", "), "\n")
}

# ── Write output with fast fwrite ─────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Metric | Before | After |
|---|---|---|
| **Rows passed to `predict()`** | Entire dataset (bug) | Only the target year's rows |
| **Subsetting speed** | `dplyr::filter` on data.frame | `data.table` keyed lookup |
| **Column assignment** | Logical vector scan each year | `data.table` in-place `:=` by key |
| **Write speed** | `write.csv` | `fwrite` (~5-10× faster) |
| **Idle memory** | Shapefile + ~15 unused libraries | Only `randomForest` + `data.table` |
| **Model integrity** | Preserved (unchanged) | Preserved (unchanged) |

The single most impactful fix is correcting the **variable masking bug**, which alone was causing the script to predict on the full dataset every iteration instead of a single year's subset. Combined with the `data.table` conversion and reduced library footprint, this should make the workflow feasible on a standard personal computer.