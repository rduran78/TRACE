 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (because the loop variable `year` shadows the column name), returning **all rows** every iteration instead of the intended subset. This means every prediction call processes the entire dataset, massively inflating computation time and memory use.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop — it consumes memory for no reason.

3. **Entire dataset held as a data.frame:** `joined_data` and `pred_db` may be large data.frames. Subsetting and assigning with base R or dplyr on hundreds of thousands of rows inside a loop is slow.

4. **All libraries loaded upfront:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `terra`, etc.) are not needed for prediction and consume memory.

5. **No chunked or batched prediction:** `predict.randomForest` on a very large `newdata` can spike memory because it must allocate the full prediction matrix across all trees simultaneously.

6. **Output written as CSV for a potentially huge table:** `write.csv` is slow for large files; `fwrite` from `data.table` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so `filter(year == yr)` works correctly |
| Unused spatial data load | Remove `st_read()` call |
| Unnecessary libraries | Load only `randomForest`, `dplyr`, `data.table` |
| Large subset + assignment in a loop | Convert `pred_db` to `data.table` and use keyed subset-assign |
| Memory spike during `predict()` | Predict in row-chunks (e.g., 50 000 rows at a time) |
| Slow CSV write | Use `data.table::fwrite()` |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(dplyr)
library(data.table)

# ── Load pre-trained models and data ──────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Objects expected in workspace after load:
#   rf_models_per_year  (named list of randomForest models)
#   joined_data         (data.frame with predictors, includes 'year')
#   pred_db             (data.frame for predictions, includes 'year')

# ── Convert to data.table for fast keyed operations ───────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Helper: chunked predict to cap memory usage ──────────────────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  # Pre-allocate output
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# ── Prediction loop (fixed variable-shadowing bug) ───────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Subset predictors for this year only (correct filter)
  test_set <- joined_data[year == yr]

  # Chunked prediction to limit peak memory
  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Fast keyed assignment

  pred_db[year == yr, consolidated := preds]

  cat(paste0("  Done — ", length(preds), " cells predicted.\n"))
}

# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

### Summary of Changes

- **Bug fix:** Renamed loop variable to `yr` so `year == yr` filters correctly, preventing full-table predictions every iteration (the single biggest performance fix).
- **Removed** the unused `st_read()` call and all unnecessary library loads, freeing significant memory.
- **Converted** to `data.table` for fast subset and assignment by reference (`:=`).
- **Added** `predict_chunked()` to break large prediction calls into 50 000-row batches, capping peak memory.
- **Replaced** `write.csv` with `fwrite` for much faster I/O.
- **Preserved** all trained Random Forest models exactly as-is — no retraining occurs.