 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Loading a large shapefile wastes significant memory.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — massively inflating computation and producing incorrect predictions.

4. **Full data frame held in memory:** `joined_data` (hundreds of thousands of rows × many predictors) is kept entirely in memory while predictions are written into `pred_db`, which may be a similarly large object.

5. **`predict.randomForest` on huge data:** Predicting on the entire (incorrectly unfiltered) dataset for every year is extremely slow and memory-intensive, since Random Forest prediction scales with `n_rows × n_trees × n_variables`.

6. **`write.csv` on a large data frame:** This is slower than alternatives like `fwrite()` from `data.table`.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read()` call |
| Variable masking bug (`year == year`) | Use a local variable with a different name (e.g., `yr`) and use `.env$yr` or base-R subsetting |
| Whole-dataframe prediction | Subset to only the current year's rows **correctly**, and select only the predictor columns needed by the model |
| Memory pressure | Use `gc()` between years; convert `joined_data` to `data.table` for faster subsetting; drop unneeded columns early |
| Slow CSV write | Use `data.table::fwrite()` |
| Optional: parallelism | Not needed once the bug is fixed and data is properly subset, but could be added later |

---

## Optimized R Code

```r
# ── Load only what is needed ──
library(randomForest)
library(data.table)

# ── Load pre-trained models ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast subsetting ──
# (joined_data and pred_db are expected to exist after loading the .RData file)
setDT(joined_data)
setDT(pred_db)

# ── Identify predictor columns from the first available model ──
first_model <- rf_models_per_year[[1]]
predictor_cols <- rownames(first_model$importance)

# ── Keep only necessary columns in joined_data to reduce memory ──
keep_cols <- unique(c("year", predictor_cols))
keep_cols <- intersect(keep_cols, names(joined_data))
joined_data <- joined_data[, ..keep_cols]
gc()

# ── Initialize the output column ──
pred_db[, consolidated := NA_real_]

# ── Predict year by year ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Correctly subset to the current year only
    test_set <- joined_data[year == yr, ..predictor_cols]

    # Predict in place
    preds <- predict(rf_model, newdata = test_set)
    pred_db[year == yr, consolidated := preds]

    # Free memory
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output efficiently ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

- **Removed** all unused library calls and the unused shapefile read — reclaims potentially gigabytes of RAM.
- **Fixed the critical bug** where `filter(year == year)` selected all rows instead of the target year, by using a distinctly named loop variable (`yr`) and `data.table` syntax (`joined_data[year == yr]`).
- **Switched to `data.table`** for fast, memory-efficient row subsetting and column selection.
- **Dropped unneeded columns** from `joined_data` early to shrink the in-memory footprint.
- **Added `gc()`** calls inside the loop to release memory between years.
- **Replaced `write.csv`** with `fwrite()` for significantly faster output.
- **Preserved all trained Random Forest models** — no retraining occurs.