 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop is comparing the column `year` to itself (always `TRUE`), so `test_set` contains *all* rows every iteration rather than only the rows for the current year. This means every year's prediction runs on the entire dataset — massively inflating computation.

2. **Loading Unnecessary Spatial Data:** `prep_data` is read via `st_read()` but is never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Using a Full `data.frame` for Prediction:** `predict.randomForest` on a `data.frame` with hundreds of thousands of rows and many columns is memory-intensive. Subsetting only the needed predictor columns reduces the footprint.

4. **Row-by-Row Assignment via Logical Indexing on a Large Data Frame:** `pred_db$consolidated[pred_db$year == year]` scans the entire data frame each iteration.

5. **Writing a Huge CSV at the End:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is significantly faster.

6. **Too Many Libraries Loaded:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, etc.) are not used during prediction and consume memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable masking in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Entire dataset predicted each year | Fix the filter so only the current year's rows are selected |
| Large data frame passed to `predict()` | Subset to only the columns the model expects |
| Slow CSV writing | Use `data.table::fwrite()` |
| Unnecessary libraries | Load only what is needed |
| Memory pressure | Use `data.table` for the main table; call `gc()` between years |

---

## Optimized Working R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data is a tibble

# ── 1. Load the pre-trained models and prediction data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: rf_models_per_year, pred_db, joined_data

# ── 2. Convert working tables to data.table for speed ──
setDT(pred_db)
setDT(joined_data)

# ── 3. Identify predictor columns the RF models expect ──
#    (taken from the first available model)
example_model <- rf_models_per_year[[1]]
predictor_cols <- rownames(example_model$importance)

# ── 4. Pre-allocate the output column ──
pred_db[, consolidated := NA_real_]

# ── 5. Create a year index on joined_data for fast subsetting ──
setkey(joined_data, year)

# ── 6. Prediction loop — fixed filter + minimal data passed ──
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {

    rf_model <- rf_models_per_year[[yr_char]]

    # Correct subsetting — only this year's rows & only needed columns
    test_set <- joined_data[year == yr, ..predictor_cols]

    # Predict in one vectorised call
    preds <- predict(rf_model, newdata = test_set)

    # Fast update by reference
    pred_db[year == yr, consolidated := preds]

    # Free memory
    rm(test_set, preds)
    gc()

  } else {
    cat(paste0("  Warning: No model found for year ", yr, "\n"))
  }
}

# ── 7. Write results efficiently ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Gains

| Before | After |
|---|---|
| All rows processed every year (filter bug) | Only the correct year's rows processed |
| ~N × Y rows predicted total (wasteful) | ~N rows predicted total |
| Full data frame with all columns sent to `predict()` | Only predictor columns sent |
| `write.csv` (slow, single-threaded) | `fwrite` (fast, multi-threaded) |
| ~15 unused libraries consuming RAM | Only 2–3 libraries loaded |
| Unused shapefile loaded into memory | Removed |

These changes should reduce both peak memory usage and total runtime by roughly an order of magnitude, making the workflow feasible on a personal computer — all while preserving the pre-trained Random Forest models exactly as-is.