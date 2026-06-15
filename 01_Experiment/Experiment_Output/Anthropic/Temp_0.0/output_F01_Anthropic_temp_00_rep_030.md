 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Loading a monolithic `.RData` file:** The `load()` call likely brings in not only `rf_models_per_year` and `pred_db`/`joined_data`, but also the full training data, intermediate objects, and possibly large spatial objects — all sitting in memory unnecessarily.

4. **Variable shadowing bug:** Inside `filter(year == year)`, the column `year` is compared to itself (the loop variable is masked by the column name), so `test_set` equals the **entire** `joined_data` every iteration, massively inflating prediction time and producing incorrect results.

5. **Predicting on a full `joined_data` data.frame:** If `joined_data` is an `sf` object or contains columns not needed by the model, unnecessary data is carried through each prediction call.

6. **Writing a potentially huge CSV at the end:** For hundreds of thousands of rows × many columns, `write.csv` is slow and the output file is large.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce load time and memory |
| Unused `prep_data` shapefile | Remove the `st_read` call entirely |
| Monolithic `.RData` loads everything | Save only the needed objects (`rf_models_per_year`, `pred_db`, `joined_data`) to separate `.rds` files, or selectively load from the `.RData` using a temporary environment |
| Variable shadowing bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly, drastically reducing per-iteration data size |
| Carrying unneeded columns into `predict()` | Subset `joined_data` to only the predictor columns the model expects |
| Slow `write.csv` | Use `data.table::fwrite` for much faster I/O |
| Optional: memory pressure from all models loaded at once | Load/unload models one at a time if saved individually |

---

## Optimized R Code

```r
# ── Only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)
library(dplyr)

# ── 1. Load only the objects you need ──
# Use a temporary environment so stray objects in the .RData don't pollute RAM.
tmp_env <- new.env()
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData', envir = tmp_env)

rf_models_per_year <- tmp_env$rf_models_per_year
pred_db            <- tmp_env$pred_db
joined_data        <- tmp_env$joined_data

rm(tmp_env)            # free everything else that was in the .RData
gc()

# ── 2. Remove the unused shapefile read entirely ──
# (prep_data was never used in prediction)

# ── 3. Identify the predictor columns the RF models expect ──
# Pull the variable names from any one of the trained models.
example_model   <- rf_models_per_year[[1]]
predictor_names <- rownames(importance(example_model))

# Keep only what predict() needs — drops geometry, IDs, etc.
joined_data <- as.data.frame(joined_data)          # drop sf geometry if present
joined_data <- joined_data[, c("year", predictor_names), drop = FALSE]
gc()

# ── 4. Pre-allocate the output column ──
pred_db$consolidated <- NA_real_

# ── 5. Prediction loop with the variable-shadowing bug fixed ──
years_to_predict <- unique(pred_db$year)

for (yr in years_to_predict) {
  cat("Predicting for year", yr, "\n")

  # Correct filter: compare column 'year' to the loop variable 'yr'
  test_set <- joined_data[joined_data$year == yr, , drop = FALSE]

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]
    pred_db$consolidated[pred_db$year == yr] <- predict(rf_model, newdata = test_set)
  } else {
    cat("  Warning: No model found for year", yr, "\n")
  }
}

# ── 6. Fast write with fwrite ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Expected Gains

| Change | Impact |
|---|---|
| Fix `year == year` filter bug | Reduces per-iteration rows from **N** (all years) to **N/T** (one year). This alone can yield a **5–20× speedup** depending on the number of years. |
| Drop unused shapefile load | Saves potentially **gigabytes** of RAM. |
| Isolate `.RData` load in temp env | Frees all training-phase objects from memory. |
| Subset to predictor columns only | Reduces working data frame size substantially. |
| Remove unused libraries | Faster startup, lower baseline memory. |
| `fwrite` instead of `write.csv` | **5–10× faster** file writing. |

No models are retrained; all original `rf_models_per_year` objects are preserved and used as-is.