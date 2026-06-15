 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — not just the target year. This massively inflates computation per iteration.

4. **Using a `data.frame` for large data:** Assigning predictions row-by-row into a large `data.frame` via conditional indexing (`pred_db$consolidated[pred_db$year == year]`) is slow for hundreds of thousands of rows.

5. **`predict.randomForest` on huge data:** With many predictor variables and a large Random Forest, prediction on hundreds of thousands of rows can spike memory. This is unavoidable per year but is worsened by the bug in point 3.

6. **Writing a massive CSV:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` spatial read | Remove it entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Slow `data.frame` operations | Convert `pred_db` and `joined_data` to `data.table` |
| Large prediction batches | Optionally chunk predictions within each year if memory is still tight |
| Slow CSV write | Use `data.table::fwrite()` |
| Keeping full `joined_data` in memory | Subset columns to only those needed by the model |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────────
library(data.table)
library(randomForest)

# ── Load the saved models and associated data ────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for speed and memory efficiency ────────────────
setDT(pred_db)
setDT(joined_data)

# ── Identify predictor columns used by the models ───────────────────────
# (Take the variable names from any one of the trained models)
example_model <- rf_models_per_year[[1]]
predictor_vars <- rownames(importance(example_model))

# Keep only the columns needed for prediction (plus 'year' for subsetting)
keep_cols <- intersect(names(joined_data), c("year", predictor_vars))
joined_data <- joined_data[, ..keep_cols]

# ── Free memory from anything else loaded by the .RData file ─────────────
invisible(gc())

# ── Initialise the output column ─────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Prediction loop (year by year) ───────────────────────────────────────
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, " ... "))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("WARNING — no model found. Skipping.\n")
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct subsetting — no variable-masking bug
  test_set <- joined_data[year == yr]

  # Optional: chunk predictions if a single year is still too large
  n <- nrow(test_set)
  chunk_size <- 50000L
  preds <- numeric(n)

  if (n <= chunk_size) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_size)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + chunk_size - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
  }

  pred_db[year == yr, consolidated := preds]

  cat(paste0(n, " cells done.\n"))

  # Free per-iteration memory
  rm(test_set, preds)
  invisible(gc())
}

# ── Write output efficiently ─────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| Change | Benefit |
|---|---|
| Removed 11 unused libraries | Lower memory baseline |
| Removed unused `st_read()` call | Avoids loading a large spatial object |
| Fixed `filter(year == year)` → `joined_data[year == yr]` | **Critical bug fix** — previously predicted on the entire dataset every iteration |
| Switched to `data.table` | Faster subsetting and in-place assignment via `:=` |
| Trimmed `joined_data` to only predictor + year columns | Reduces memory footprint substantially |
| Added chunked prediction option | Caps peak memory during `predict()` |
| Replaced `write.csv` with `fwrite` | Much faster I/O |
| Added `gc()` calls per iteration | Returns freed memory to the OS promptly |

No models are retrained; only the prediction pipeline is optimized.