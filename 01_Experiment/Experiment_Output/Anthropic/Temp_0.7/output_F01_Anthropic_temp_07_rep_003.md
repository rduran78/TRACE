 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial shapefile (`st_read`) that is never used** in the prediction loop. This can consume a huge amount of RAM for no reason.

3. **Variable shadowing bug:** `filter(year == year)` inside the loop does not filter correctly because the loop variable `year` shadows the column name `year`. This likely returns the entire dataset each iteration, causing redundant and massively inflated prediction work.

4. **Predicting on `joined_data` but writing to `pred_db`:** The code filters `joined_data` for the test set but assigns predictions into `pred_db`. If these are different-sized objects or have different row orderings, results will be incorrect or will error out. This also means both large objects must coexist in memory.

5. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive because it internally builds a matrix of all tree predictions. Doing this in a single call per year with no chunking can spike memory.

6. **Writing a massive CSV with `write.csv`:** For very large data frames, `write.csv` is slow. `data.table::fwrite` is dramatically faster.

7. **The entire `pred_db` data frame is kept in memory** alongside `joined_data` and the full list of RF models, which compounds memory pressure.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused shapefile load | Remove `st_read` call entirely |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Both `joined_data` and `pred_db` in memory | Confirm they share the same structure; remove `joined_data` if `pred_db` is the prediction target, or unify them |
| Large single-call `predict()` | Chunk predictions into batches (e.g., 50,000 rows) to limit peak memory |
| Slow `write.csv` | Use `data.table::fwrite` |
| All RF models in memory at once | Extract one model at a time and free memory with `gc()` after each year |

---

## Optimized R Code

```r
# ── Load only what is needed ──
library(randomForest)
library(dplyr)
library(data.table)

# ── Load pre-trained models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# If joined_data is the source of predictors and pred_db is the output target,
# make sure they are row-aligned. If they are the same table, unify to save RAM:
# pred_db <- joined_data   # uncomment if appropriate

pred_db$consolidated <- NA_real_

# ── Chunked prediction function to control peak memory ──
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, , drop = FALSE])
  }
  return(preds)
}

# ── Prediction loop (with corrected filter) ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  if (as.character(yr) %in% names(rf_models_per_year)) {

    # Correct filter: use the loop variable 'yr', not 'year'
    row_idx <- which(pred_db$year == yr)
    test_set <- joined_data[row_idx, , drop = FALSE]

    rf_model <- rf_models_per_year[[as.character(yr)]]

    pred_db$consolidated[row_idx] <- predict_chunked(rf_model, test_set)

    # Free the individual model copy and force garbage collection
    rm(rf_model, test_set)
    gc()

  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Optional: free the full model list if no longer needed ──
rm(rf_models_per_year, joined_data)
gc()

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed ~10 unused libraries | Lower memory baseline |
| 2 | Removed unused `st_read` shapefile load | Potentially saves GBs of RAM |
| 3 | Renamed loop variable to `yr` | Fixes the critical variable-shadowing bug in `filter()`/`which()` |
| 4 | Added `predict_chunked()` | Caps peak memory during prediction |
| 5 | Added `rm()` + `gc()` inside loop | Frees each year's model and test set immediately after use |
| 6 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster I/O |
| 7 | **No models were retrained** | Requirement preserved |

These changes should make the script feasible on a normal machine with 8–16 GB of RAM.