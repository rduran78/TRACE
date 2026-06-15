 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on *all* rows, massively inflating computation.

3. **Entire `joined_data` held in memory:** If `joined_data` has hundreds of thousands of rows and many predictor columns, keeping it as a single in-memory `data.frame`/`tibble` alongside `pred_db`, `prep_data`, and all the RF models simultaneously can exhaust RAM.

4. **`predict.randomForest` on huge data:** Even with a correct subset, calling `predict()` on hundreds of thousands of rows at once can spike memory because `randomForest` internally allocates matrices proportional to `n_rows × n_trees`.

5. **Heavy library loading:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, `zoo`) are loaded but unused in the prediction phase, consuming memory.

6. **Writing a potentially huge CSV with `write.csv`:** For very large data, `write.csv` is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Remove the unused `st_read()` call and unused libraries | Free hundreds of MB of RAM |
| 2 | Fix the variable-shadowing bug (`year == year`) | Reduce each iteration's data from N to N/num_years |
| 3 | Convert `joined_data` to a `data.table` and subset by reference | `data.table` subsetting is faster and more memory-efficient than `dplyr::filter` |
| 4 | Batch the `predict()` call in chunks if a single year is still too large | Caps peak memory inside `randomForest:::predict` |
| 5 | Remove columns from `joined_data` that are not needed by the RF models | Fewer columns → smaller working set |
| 6 | Use `data.table::fwrite` instead of `write.csv` | Much faster serialization |
| 7 | Call `gc()` after each year to release memory promptly | Helps on RAM-constrained machines |

The trained Random Forest models are **preserved exactly as-is**; nothing is retrained.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(data.table)
library(randomForest)   # needed for predict()

# ── Load the saved workspace (contains rf_models_per_year, joined_data, pred_db) ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Remove prep_data if it was loaded; it is not used ──
if (exists("prep_data")) rm(prep_data)
gc()

# ── Convert to data.table for fast, memory-efficient operations ──
setDT(joined_data)
setDT(pred_db)

# ── Identify predictor columns expected by the RF models ──
# (use the first available model to discover them)
example_model <- rf_models_per_year[[1]]
needed_cols   <- rownames(importance(example_model))
keep_cols     <- intersect(c("year", needed_cols), names(joined_data))
joined_data   <- joined_data[, ..keep_cols]    # drop unneeded columns
gc()

# ── Initialise the output column ──
pred_db[, consolidated := NA_real_]

# ── Helper: chunked predict to cap peak memory ──
chunked_predict <- function(model, newdata, chunk_size = 50000L) {
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

# ── Prediction loop (one year at a time) ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── Correct subset: use the loop variable, not the column name ──
  test_set <- joined_data[year == yr]

  # ── Predict in chunks to limit peak RAM ──
  preds <- chunked_predict(rf_model, test_set, chunk_size = 50000L)


  pred_db[year == yr, consolidated := preds]

  # ── Free memory after each year ──
  rm(test_set, preds, rf_model)
  gc()
}

# ── Fast write ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

### Summary of Key Changes

* **Bug fix:** Renamed the loop variable to `yr` so the filter `year == yr` correctly subsets a single year instead of returning all rows.
* **Memory reduction:** Removed the unused shapefile read, dropped unneeded libraries, trimmed `joined_data` to only the columns the models require, and added `gc()` calls.
* **Speed improvement:** Switched to `data.table` for subsetting and writing, and added optional chunked prediction to avoid a single massive allocation inside `predict.randomForest`.
* **Models untouched:** No retraining occurs; the pre-trained `rf_models_per_year` list is used directly.