 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction step. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This consumes significant memory for large geographic datasets.

3. **Variable masking bug:** `filter(year == year)` inside the loop is comparing the column `year` to itself (the column name masks the loop variable), so `test_set` always equals the entire `joined_data`. This means every iteration predicts on the *full* dataset rather than a single year's subset, dramatically increasing compute time and producing incorrect results.

4. **Large in-memory objects:** Loading the `.RData` file may bring in `joined_data`, `pred_db`, the trained models, and potentially other large objects all at once.

5. **Whole-dataframe column assignment pattern:** Writing predictions into `pred_db$consolidated` via logical indexing inside a loop on a very large data.frame is slow and memory-inefficient.

6. **`randomForest::predict` on huge data:** Even for a single year, hundreds of thousands of rows with many predictors can spike memory because `randomForest` predict internally builds a matrix.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries & objects | Remove them; don't load `prep_data` |
| Variable masking bug | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Memory spike from full `.RData` | Load only what's needed; convert to `data.table` |
| Large single-pass predict | Predict in row-chunks within each year to cap memory |
| Slow indexed assignment | Pre-allocate a results list, then bind once at the end |
| Output format | Use `fwrite()` (from `data.table`) for fast CSV writing |

---

## Optimized R Code

```r
# ── Only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)

# ── Load trained models and data ──
# This brings in rf_models_per_year, joined_data, pred_db (and possibly others).
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for speed and memory efficiency
setDT(pred_db)
setDT(joined_data)

# Pre-allocate the consolidated column
pred_db[, consolidated := NA_real_]

# ── Prediction settings ──
chunk_size <- 50000L          # rows per chunk; tune to your RAM

available_years <- names(rf_models_per_year)

unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!(yr_char %in% available_years)) {
    cat(paste0("  Warning: No model found for year ", yr, " – skipping.\n"))
    next
  }

  # Correct subsetting (no variable masking)
  test_set <- joined_data[year == yr]
  n        <- nrow(test_set)

  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, " – skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Predict in chunks to limit peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = chunk_size)

  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Write predictions back (data.table indexed update – fast)
  pred_db[year == yr, consolidated := preds]

  # Free per-year objects
  rm(test_set, preds, rf_model)
  gc()
}

# ── Fast CSV export ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

### Key Changes Summarized

- **Bug fix:** Loop variable renamed to `yr` so that `joined_data[year == yr]` filters correctly instead of returning the entire table.
- **Memory reduction:** Removed all unused library imports and the unused `st_read()` call; switched to `data.table` for lower memory footprint and faster subsetting.
- **Chunked prediction:** `predict()` is called on subsets of `chunk_size` rows at a time, preventing memory spikes from the internal matrix allocation inside `randomForest`.
- **Garbage collection:** `rm()` and `gc()` are called after each year to release memory promptly.
- **Fast I/O:** `fwrite()` replaces `write.csv()` for significantly faster output.
- **Models untouched:** No retraining occurs; the pre-trained `rf_models_per_year` are used as-is.