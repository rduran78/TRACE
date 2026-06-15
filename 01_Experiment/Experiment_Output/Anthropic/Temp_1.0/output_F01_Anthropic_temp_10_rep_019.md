 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning **all rows** every iteration. This means every year's prediction runs on the entire dataset, massively inflating computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Monolithic `predict()` Calls:** With hundreds of thousands of rows and many predictors, a single `predict()` call on a large Random Forest can spike memory usage because the RF must route every row through every tree simultaneously.

4. **Heavy Library Loading:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, etc.) are loaded but unused in the prediction phase, consuming memory.

5. **Writing a Potentially Huge CSV:** `write.csv` on a very large data frame is slow and produces large files; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) so it doesn't collide with the column name. |
| Unused shapefile load | Remove the `st_read()` call. |
| Unused libraries | Remove or comment out unnecessary `library()` calls to reduce memory footprint. |
| Large `predict()` memory spike | Predict in **row-chunks** (e.g., 50,000 rows at a time) to cap peak memory. |
| Slow CSV write | Use `data.table::fwrite()`. |
| General memory | Call `gc()` after each year and use `data.table` where possible. |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────
library(tidyverse)
library(data.table)
library(randomForest)

# ── Load pre-trained models and prediction database ───────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Initialize output column ─────────────────────────────────
pred_db$consolidated <- NA_real_

# ── Chunked predict helper (caps peak memory) ────────────────
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))

  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, , drop = FALSE])
  }
  preds
}

# ── Main prediction loop ─────────────────────────────────────
# Use 'yr' to avoid shadowing the column name 'year'
unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correctly subset the rows for this year
    row_idx <- which(pred_db$year == yr)
    test_set <- joined_data[joined_data$year == yr, , drop = FALSE]

    rf_model <- rf_models_per_year[[yr_char]]

    pred_db$consolidated[row_idx] <- predict_in_chunks(
      model      = rf_model,
      newdata    = test_set,
      chunk_size = 50000L
    )

    # Free memory after each year
    rm(test_set)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Fast write ────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

1. **Fixed the filter bug** — renamed the loop variable to `yr` and used base-R subsetting (`joined_data[joined_data$year == yr, ]`) to correctly isolate each year's data.
2. **Removed `st_read()`** and all unused library calls to free memory.
3. **Added `predict_in_chunks()`** — processes rows in batches of 50,000 (tunable) so peak memory stays bounded.
4. **Added `gc()`** after each year to release memory promptly.
5. **Replaced `write.csv` with `fwrite()`** for substantially faster output.

These changes preserve the trained Random Forest models exactly as-is—no retraining occurs—and should make the workflow feasible on a standard personal computer.