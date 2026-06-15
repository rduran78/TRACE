 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Scoping Bug in `filter()`:** `filter(year == year)` is comparing the column `year` to itself (always `TRUE`), so `test_set` contains the *entire* dataset every iteration — not a single year's subset. This means the Random Forest `predict()` call processes the full dataset every loop iteration, massively inflating computation.

2. **Loading a Large Shapefile Unnecessarily:** `prep_data` is loaded via `st_read()` but is never used in the prediction loop. Reading a large `.shp` file consumes substantial memory for no purpose.

3. **Whole-Dataframe Assignment in a Loop:** Writing predictions back into `pred_db` row-by-row (year-by-year) using `pred_db$consolidated[pred_db$year == year]` forces repeated logical subsetting over a very large data frame each iteration.

4. **Many Unused Libraries:** Libraries such as `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, and `zoo` are loaded but appear unused, consuming memory.

5. **Using `data.frame` Instead of `data.table`:** For hundreds of thousands of rows and many columns, base R data frames and `tidyverse` verbs are slower than `data.table` operations.

6. **`write.csv` on a Large Data Frame:** This is slow; `fwrite()` from `data.table` is dramatically faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Scoping bug in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Unused libraries | Remove them to free memory |
| Slow subsetting & assignment | Convert to `data.table` and use keyed joins or `:=` by reference |
| Slow CSV write | Use `data.table::fwrite()` |
| Memory pressure | Call `gc()` after large operations; only load what is needed from the `.RData` file |

---

## Optimized Working R Code

```r
# ── Load only necessary libraries ──────────────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)       # kept only if joined_data is a tibble

# ── Load pre-trained models and prediction database ────────────────
# (Do NOT load the shapefile — it is unused in prediction)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed ───────────────────────────────
setDT(pred_db)
setDT(joined_data)

# Initialize the output column
pred_db[, consolidated := NA_real_]

# Key both tables by year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# ── Prediction loop (year by year) ────────────────────────────────
unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correctly subset to the current year only
    test_set <- joined_data[.(yr)]

    rf_model <- rf_models_per_year[[yr_char]]

    # Predict and assign by reference (no copy)
    pred_db[.(yr), consolidated := predict(rf_model, newdata = test_set)]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each year
  gc()
}

# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

1. **Fixed the critical `filter(year == year)` bug** by renaming the loop variable to `yr` and using `data.table` keyed subsetting `[.(yr)]`, which correctly selects only the rows for that year.
2. **Removed the unused `st_read()` call**, freeing potentially gigabytes of memory.
3. **Removed all unused library loads** (`blockCV`, `zoo`, `LongituRF`, `sf`, `spdep`, `terra`, `plm`, `fixest`, `scales`, `stringi`, `utils`).
4. **Converted to `data.table`** with keyed subsetting and `:=` assignment by reference — avoids copying the entire data frame on every iteration.
5. **Replaced `write.csv` with `fwrite()`** for significantly faster output.
6. **Added `gc()`** after each year's prediction to reclaim memory promptly.

These changes preserve all trained Random Forest models exactly as they are (no retraining) and should make the workflow feasible on a standard personal computer.