 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` contains **all rows** every iteration instead of just one year's data. This causes every prediction call to run on the full dataset, massively inflating computation time.

3. **Entire `joined_data` held in memory:** If `joined_data` has hundreds of thousands of rows × many predictor columns, keeping it as a single `tibble`/`data.frame` in memory alongside `pred_db` and `prep_data` is expensive.

4. **`predict.randomForest` on huge data in one call:** Even after fixing the filter bug, predicting on hundreds of thousands of rows at once can spike memory because `randomForest::predict` internally allocates matrices proportional to `n_rows × n_trees`.

5. **Writing a potentially enormous CSV with `write.csv`:** For very large data frames, `write.csv` is slow; `data.table::fwrite` is significantly faster.

6. **Many unused libraries loaded:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra` are loaded but unused, consuming memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `prep_data` load | Remove the `st_read()` call |
| Unused libraries | Remove all libraries not needed for prediction |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` |
| Memory from full `joined_data` | Convert to `data.table`; subset by year efficiently; drop unneeded columns |
| Large single `predict()` call | Chunk predictions into batches (e.g., 50 000 rows) to cap peak memory |
| Slow `write.csv` | Use `data.table::fwrite` |
| Optional: garbage collection | Call `gc()` after each year to free memory between iterations |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(data.table)
library(dplyr)            # only for the pipe / filter if needed

# ── Load pre-trained models and data ──────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring in: rf_models_per_year, joined_data, pred_db

# ── Remove any large objects that were loaded but are not needed ──
# (Adjust names to whatever else the .RData file contains)
# rm(prep_data)  # not needed
gc()

# ── Convert to data.table for fast subset / assign ────────────────
setDT(joined_data)
setDT(pred_db)

# ── Identify predictor columns once (from any model) ─────────────
sample_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(importance(sample_model))

# ── Keep only necessary columns in joined_data to save memory ─────
keep_cols <- intersect(c("year", predictor_vars), names(joined_data))
joined_data <- joined_data[, ..keep_cols]
gc()

# ── Initialise the output column ──────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Helper: chunked predict to limit peak memory ─────────────────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# ── Prediction loop (fixed filter + chunked predict) ─────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

 if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct, non-shadowed subset
  test_set <- joined_data[year == yr, ..predictor_vars]

  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  pred_db[year == yr, consolidated := preds]

  # Free memory between years
  rm(test_set, preds)
  gc()
}

# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

### Summary of Key Changes

- **Bug fix:** Renamed loop variable to `yr` so `joined_data[year == yr]` correctly subsets a single year instead of returning the entire dataset every iteration.
- **Memory reduction:** Removed unused library loads and the unused `st_read` call; trimmed `joined_data` to only the required predictor columns; added `gc()` calls between years.
- **Chunked prediction:** `predict_chunked()` processes rows in configurable batches (default 50,000) so that `randomForest::predict` never needs to allocate an `n × n_trees` matrix for the full dataset at once.
- **Faster I/O:** Replaced `write.csv` with `data.table::fwrite`.
- **No model retraining:** All pre-trained `rf_models_per_year` objects are preserved and used as-is.