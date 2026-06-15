 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Loading a large shapefile wastes significant memory.

3. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` always equals the entire `joined_data` dataframe. This means every iteration predicts on *all* rows, not just the target year — massively inflating computation time and producing incorrect results.

4. **Monolithic predict call:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows at once for a large Random Forest can spike memory usage because `randomForest::predict` builds a full matrix of tree-level predictions internally.

5. **Full dataframe written with `write.csv`:** `write.csv` on a very large dataframe is slow; `data.table::fwrite` is much faster.

6. **Loading the entire `.RData` workspace:** This may pull in large objects (like `joined_data`, `pred_db`, and all year-specific RF models simultaneously), consuming memory even when only one model is needed at a time. However, since the models are stored in a single `.RData` file, we must load them all — but we can free each model after use.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` |
| Memory spike during `predict()` | Predict in chunks (batches) and garbage-collect between years |
| Slow CSV write | Use `data.table::fwrite` |
| All RF models in memory simultaneously | Free each model after its year is done with `rm()` + `gc()` |
| `joined_data` kept fully in memory | Convert to `data.table` for faster subsetting |

---

## Optimized R Code

```r
# ── Only load what is actually needed ──
library(randomForest)
library(data.table)

# ── Load trained models and prediction data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year (named list), joined_data, pred_db

# ── Convert to data.table for fast operations ──
setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# ── Chunked prediction function to limit memory ──
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Prediction loop (fixed filter + memory management) ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Subset the features for this year
    test_set <- joined_data[year == yr]

    # Retrieve model, predict in chunks, then free the model
    rf_model <- rf_models_per_year[[yr_char]]
    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

    pred_db[year == yr, consolidated := preds]

    # Free memory for this year's model and data
    rf_models_per_year[[yr_char]] <- NULL
    rm(rf_model, test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Clean up remaining objects ──
rm(rf_models_per_year, joined_data)
gc()

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

### Key changes summarized:

- **Bug fix:** Loop variable renamed to `yr` so `joined_data[year == yr]` filters correctly.
- **Memory:** Each RF model is nullified from the list after use; `gc()` is called each iteration.
- **Chunked prediction:** Rows are predicted in batches of 50,000 to avoid a single massive internal matrix allocation.
- **Fast I/O:** `fwrite` replaces `write.csv`.
- **Removed all unused libraries and the unused shapefile load**, reclaiming substantial memory.