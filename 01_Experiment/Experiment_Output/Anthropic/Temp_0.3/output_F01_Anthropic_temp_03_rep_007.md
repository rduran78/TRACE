 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geographic cells.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — massively inflating computation and producing incorrect predictions.

4. **Using `predict()` on the full `joined_data` tibble:** Random Forest `predict()` on a large `tibble`/`data.frame` with many unused columns is slower and more memory-intensive than necessary. Only the predictor columns should be passed.

5. **Writing a potentially huge CSV with `write.csv`:** For very large data, `data.table::fwrite` is far faster and more memory-efficient.

6. **No garbage collection or memory management:** The `.RData` file likely contains the training data (`joined_data`), all models, and other objects. These all persist in memory simultaneously.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused library calls | Reduce memory footprint |
| Remove unused `prep_data` shapefile read | Free potentially gigabytes of RAM |
| Fix the `year == year` variable masking bug | Correctness — also prevents predicting on the entire dataset each iteration |
| Extract only required predictor columns for `predict()` | Reduce memory passed to the model |
| Use `data.table::fwrite` instead of `write.csv` | Much faster I/O for large files |
| Call `gc()` after removing unneeded objects | Reclaim memory |
| Process predictions in chunks if needed | Prevent memory spikes |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)
library(dplyr)

# ── Load the saved models and associated data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load the shapefile — it is unused in prediction.
# prep_data <- st_read(...)  # REMOVED

# ── Identify the predictor variable names from one of the trained models ──
sample_model <- rf_models_per_year[[names(rf_models_per_year)[1]]]
predictor_vars <- rownames(importance(sample_model))

# ── Remove any large objects from the .RData that are not needed ──
# Keep only: pred_db, joined_data, rf_models_per_year
# (Adjust object names if the .RData contains other large objects)
objects_to_keep <- c("pred_db", "joined_data", "rf_models_per_year",
                     "predictor_vars", "sample_model")
rm(list = setdiff(ls(), objects_to_keep))
rm(sample_model)
gc()

# ── Convert joined_data to data.table for faster subsetting ──
setDT(joined_data)
setDT(pred_db)

# ── Initialise the output column ──
pred_db[, consolidated := NA_real_]

# ── Prediction loop with the variable-masking bug fixed ──
years_to_predict <- unique(pred_db$year)

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Subset only the rows for this year and only the predictor columns
    test_set <- joined_data[year == yr, ..predictor_vars]

    # Predict
    preds <- predict(rf_model, newdata = test_set)

    # Assign predictions back
    pred_db[year == yr, consolidated := preds]

    # Free memory from this iteration
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

| Original Issue | Fix Applied |
|---|---|
| `filter(year == year)` selects all rows (masking bug) | Renamed loop variable to `yr`; used `data.table` syntax `joined_data[year == yr, ...]` |
| Full `joined_data` passed to `predict()` | Only predictor columns (`..predictor_vars`) are passed |
| Unused shapefile loaded into memory | Removed entirely |
| 14 libraries loaded; most unused | Reduced to 3 (`randomForest`, `data.table`, `dplyr`) |
| `write.csv` on large data | Replaced with `data.table::fwrite` |
| No memory management | Added `rm()` and `gc()` calls after each year and after setup |

These changes preserve all trained Random Forest models exactly as-is and require no retraining.