 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary heavy library loading:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`prep_data`) that is never used** in the prediction section. This can consume gigabytes of RAM for no reason.

3. **Variable shadowing bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which means it does **not** filter correctly — it evaluates to `TRUE` for every row, so `test_set` equals the entire `joined_data` every iteration. This causes the model to predict on the full dataset every year, massively inflating computation time and producing wrong results.

4. **Predicting on the full dataframe at once** for each year can spike memory, especially with a large Random Forest model and hundreds of thousands of rows with many predictors.

5. **Writing the entire result with `write.csv`** is slower than alternatives like `fwrite` from `data.table`.

6. **No garbage collection or memory management** between yearly iterations.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries and objects | Remove them or don't load them |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Full-data prediction per iteration | Fix the filter so only the relevant year's rows are predicted |
| Memory pressure during prediction | Predict in chunks if needed; call `gc()` between years |
| Slow CSV write | Use `data.table::fwrite()` |
| Storing predictions | Use `data.table` for fast row-subset assignment |

---

## Optimized R Code

```r
# ── Load only what is needed ──
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data / pred_db are tibbles

# ── Load pre-trained models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load the shapefile — it is unused in prediction.
# prep_data <- st_read(...)   # REMOVED

# ── Convert to data.table for fast subset assignment ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Get the years that actually have a trained model ──
available_years <- names(rf_models_per_year)
unique_years    <- as.character(unique(pred_db$year))
years_to_predict <- intersect(unique_years, available_years)

cat(paste0("Years with models: ", paste(years_to_predict, collapse = ", "), "\n"))

# ── Predict year-by-year ──
for (yr in years_to_predict) {

  cat(paste0("Predicting for year ", yr, " ... "))

  # Correct filter — no variable shadowing
  test_set <- joined_data[year == as.numeric(yr)]

  if (nrow(test_set) == 0L) {
    cat("no data, skipping.\n")
    next
  }

  rf_model <- rf_models_per_year[[yr]]

  # --- Optional: chunk-based prediction to limit peak memory ---
  chunk_size <- 50000L
  n          <- nrow(test_set)
  preds      <- numeric(n)

  if (n <= chunk_size) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_size)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + chunk_size - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
  }

  # Fast assignment by reference
  pred_db[year == as.numeric(yr), consolidated := preds]

  cat(paste0(n, " cells done.\n"))

  # Free memory
  rm(test_set, preds)
  gc()
}

# Report years without a model
missing <- setdiff(unique_years, available_years)
if (length(missing) > 0) {
  cat(paste0("Warning: No model for year(s): ", paste(missing, collapse = ", "), "\n"))
}

# ── Write output efficiently ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Removed** unused libraries and the unused shapefile load → frees GBs of RAM.
- **Fixed the critical filter bug** (`year == year` → `year == as.numeric(yr)`) so each iteration only processes the correct subset.
- **Switched to `data.table`** for fast by-reference column assignment and subsetting.
- **Added chunked prediction** (configurable `chunk_size`) to cap peak memory during `predict()`.
- **Added `gc()`** after each year to reclaim memory.
- **Replaced `write.csv` with `fwrite`** for significantly faster output.
- **Did not retrain** any model — all original `rf_models_per_year` objects are used as-is.