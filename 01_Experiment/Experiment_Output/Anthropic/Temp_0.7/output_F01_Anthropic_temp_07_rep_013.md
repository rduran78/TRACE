 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary heavy libraries loaded:** `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Unnecessary spatial data read:** `prep_data` is read from a shapefile but never used. Loading a large `.shp` file into memory is wasteful.

3. **Loading a potentially massive `.RData` file entirely into memory:** The `.RData` file likely contains training data (`joined_data`), the prediction database (`pred_db`), and the models (`rf_models_per_year`). The training data is no longer needed for prediction but remains in memory.

4. **Variable masking bug in the loop:** `filter(year == year)` is comparing the column `year` to itself (always `TRUE`) because the loop variable `year` is masked by the column name. This means `test_set` contains **all rows** every iteration, not just the rows for that year. This causes the prediction to run on the entire dataset for every year — dramatically increasing computation time and producing incorrect results.

5. **Using a full `data.frame` for prediction:** `randomForest::predict` on hundreds of thousands of rows in a single call can spike memory. No chunking is used.

6. **Writing a massive CSV at the end:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Training data in memory | Remove `joined_data` (and any other unneeded objects) from the environment after loading `.RData`, then call `gc()` |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly |
| Large single prediction call | Process predictions in chunks (e.g., 50,000 rows) to keep peak memory bounded |
| Slow `write.csv` | Use `data.table::fwrite` |
| General memory | Call `gc()` after removing large objects and between years |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(tidyverse)
library(data.table)

# ── Load saved models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Free objects that are not needed for prediction ──
# (Adjust object names if your .RData contains different names)
if (exists("prep_data"))    rm(prep_data)
if (exists("joined_data"))  rm(joined_data)
# Remove any other large training-phase objects here, e.g.:
# if (exists("train_set")) rm(train_set)
gc()

# ── Confirm required objects exist ──
stopifnot(exists("pred_db"), exists("rf_models_per_year"))

# ── Initialise the output column ──
pred_db$consolidated <- NA_real_

# ── Chunked prediction function (controls peak memory) ──
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, , drop = FALSE])
  }
  preds
}

# ── Prediction loop (note: loop variable is 'yr' to avoid masking) ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, " ... "))

  year_key <- as.character(yr)


  if (!year_key %in% names(rf_models_per_year)) {
    cat("WARNING – no model found, skipping.\n")
    next
  }

  # Subset only the rows for this year
  idx <- which(pred_db$year == yr)
  test_set <- pred_db[idx, , drop = FALSE]

  # Predict in chunks to limit memory spikes
  rf_model <- rf_models_per_year[[year_key]]
  pred_db$consolidated[idx] <- predict_chunked(rf_model, test_set)

  # Tidy up within the loop
  rm(test_set)
  gc()

  cat(paste0(length(idx), " cells done.\n"))
}

# ── Write output (fwrite is ~5-10× faster than write.csv) ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done. Output written to RF_imputated_db.csv\n")
```

---

### Summary of Changes

- **Removed** 10 unused library calls and the unused `st_read`.
- **Fixed the critical bug:** renamed loop variable to `yr` so `pred_db$year == yr` filters correctly instead of returning all rows every iteration.
- **Freed training-phase objects** immediately after loading, reclaiming potentially gigabytes of RAM.
- **Added chunked prediction** (`predict_chunked`) so that no single `predict()` call operates on more than 50,000 rows at a time, bounding peak memory.
- **Replaced `write.csv` with `fwrite`** for substantially faster I/O.
- **Added `gc()` calls** between years to return freed memory to the OS promptly.

These changes preserve all trained Random Forest models untouched and should make the workflow feasible on a standard personal computer.