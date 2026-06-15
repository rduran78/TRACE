 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr::filter` to evaluate `year == year` as `TRUE` for every row (self-comparison), meaning `test_set` contains **all rows** every iteration — not just the target year. This massively inflates computation per iteration.

4. **Using a `data.frame` for large row-wise assignment:** Assigning predictions back into `pred_db$consolidated` row by row (per year) on a very large data frame is slow due to R's copy-on-modify semantics.

5. **`predict.randomForest` on hundreds of thousands of rows at once** can spike memory because it internally allocates matrices across all trees.

6. **Writing a massive CSV at the end** with `write.csv` is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| `prep_data` never used | Do not load it |
| Variable masking bug in `filter()` | Rename the loop variable (e.g., `yr`) so it doesn't collide with the column name |
| Slow data.frame assignment | Convert to `data.table` and assign by reference with `:=` |
| Memory spike during `predict()` | Predict in chunks (batches) within each year |
| Slow `write.csv` | Use `data.table::fwrite` |

---

## Optimized R Code

```r
# ── Load only what is needed ──
library(randomForest)
library(data.table)
library(dplyr)          # only for the initial filter; could also use data.table

# ── Load the pre-trained models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for fast by-reference operations ──
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Batch-predict helper (controls peak memory) ──
predict_in_batches <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Main prediction loop ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correct filter: use the loop variable 'yr', not 'year'
  test_set <- joined_data[year == yr]

  model_key <- as.character(yr)

  if (model_key %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[model_key]]

    preds <- predict_in_batches(rf_model, test_set, batch_size = 50000L)

    # Assign by reference — no copy of the entire table
    pred_db[year == yr, consolidated := preds]

    # Free memory immediately
    rm(preds, test_set, rf_model)
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

- **Removed** all unused library calls and the unused `st_read()` call, reclaiming significant memory.
- **Fixed the critical bug:** renamed the loop iterator to `yr` so that `year == yr` correctly subsets to a single year instead of returning the entire dataset.
- **Switched to `data.table`:** uses `:=` for zero-copy column assignment and `[year == yr]` for fast subsetting.
- **Added batch prediction:** `predict_in_batches()` splits large year-groups into chunks of 50,000 rows (tunable), preventing memory spikes from `predict.randomForest` allocating a full n × n_trees matrix at once.
- **Explicit `gc()` after each year:** frees memory from the previous iteration's predictions and test set.
- **Replaced `write.csv` with `fwrite`:** typically 10–50× faster for large files.

No models are retrained; only the prediction and data-handling pipeline is optimized.