 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several issues causing poor performance:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column to itself (always TRUE), so every iteration loads the **entire** dataset instead of just one year's subset. This means every `predict()` call processes *all* rows, not just one year's worth.

2. **Unnecessary spatial/heavy libraries loaded:** Libraries like `sf`, `spdep`, `terra`, `blockCV`, `LongituRF`, `plm`, `fixest` are not needed for prediction and consume memory.

3. **Full data loaded into memory at once:** `joined_data` (hundreds of thousands of rows × many predictors × many years) stays fully resident in memory while predictions are made.

4. **`predict.randomForest` on massive data frames is RAM-intensive:** Random Forest prediction materialises intermediate matrices proportional to `n_rows × n_trees`.

5. **Growing a data.frame inside a loop via column assignment** (`pred_db$consolidated[...]`) on a very large data.frame is not ideal but is secondary to the above.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | **Fix the variable-shadowing bug** — rename the loop variable (e.g., `yr`) so the `dplyr::filter` actually subsets to one year. | Eliminates predicting on the full dataset every iteration. |
| 2 | **Drop unused libraries** to free memory. | Reduces baseline RAM footprint. |
| 3 | **Convert to `data.table`** for faster subsetting and in-place assignment. | `data.table` is more memory-efficient than tibble/data.frame for large data. |
| 4 | **Predict in chunks** within each year if a single year is still too large. | Caps peak memory used by `predict()`. |
| 5 | **Explicitly remove objects and call `gc()`** after each year. | Returns memory to the OS between iterations. |
| 6 | **Write results incrementally** (append to CSV per year) so partial results are not lost. | Safer and avoids holding everything in memory until the end. |

---

## Optimized R Code

```r
# ── Only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)

# ── 1. Load the pre-trained models ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Convert prediction source to data.table (fast subset / update) ──
#    (joined_data is the table that contains the predictor columns)
setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# ── 3. Prepare an output CSV (write header once) ──
out_file <- "RF_imputed_db.csv"
first_write <- TRUE

# ── 4. Helper: chunked predict to cap memory ──
chunk_predict <- function(model, newdata, chunk_size = 50000L) {
  n   <- nrow(newdata)
  out <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    out[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  out
}

# ── 5. Year-by-year prediction loop ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s …\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # Subset only this year's rows (fixed shadowing bug)
  test_idx  <- which(joined_data$year == yr)
  test_set  <- joined_data[test_idx]

  rf_model  <- rf_models_per_year[[yr_char]]

  # Chunked prediction to limit peak RAM
  preds <- chunk_predict(rf_model, test_set, chunk_size = 50000L)

  # In-place update in pred_db
  pred_db[year == yr, consolidated := preds]

  # ── 6. Write this year's results incrementally ──
  year_rows <- pred_db[year == yr]
  fwrite(year_rows, file = out_file, append = !first_write)
  first_write <- FALSE

  # Free memory

  rm(test_set, preds, year_rows, rf_model)
  gc()

  cat(sprintf("  Done – %d rows predicted.\n", length(test_idx)))
}

cat("All years complete. Results in:", out_file, "\n")
```

---

## Summary of Key Changes

- **Bug fix:** Loop variable renamed to `yr` so `joined_data$year == yr` filters correctly.
- **Memory savings:** Unused libraries removed; chunked `predict()`; explicit `rm()` + `gc()` per iteration; incremental CSV writing.
- **Speed gains:** `data.table` for O(1)-style row subsetting and in-place `:=` column updates; predicting only the correct year subset (not the full table).
- **Models untouched:** No retraining; `rf_models_per_year` is loaded and used as-is.