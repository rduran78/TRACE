 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` equals the **entire** `joined_data` every iteration. This is both incorrect and extremely expensive.

4. **Whole-dataframe prediction per iteration:** Even if the filter worked correctly, calling `predict()` on hundreds of thousands of rows with a large Random Forest model is memory-intensive. There is no chunking.

5. **Using a `data.frame` for row-level assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` repeatedly on a very large data frame is slow due to repeated logical indexing and potential memory copying.

6. **`joined_data` vs `pred_db` ambiguity:** The prediction features come from `joined_data`, but results are stored in `pred_db`. If these are separate large objects both in memory, that doubles RAM usage.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Entire dataset loaded every iteration | Fix the filter so only one year's data is processed |
| Large predict calls | Chunk predictions within each year if still too large |
| Slow data.frame assignment | Convert to `data.table` for fast update-by-reference |
| Two large objects in memory | Unify or minimize duplication; use `gc()` after large operations |
| Writing a huge CSV | Use `data.table::fwrite()` for speed |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(randomForest)
library(data.table)

# ── Load the saved models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove any large objects loaded from the .RData that are NOT needed
# (Adjust object names based on what is actually in the .RData file.)
# For example, if prep_data or other training objects were saved:
# rm(prep_data, training_data, ...)
gc()

# ── Convert both data frames to data.table for speed ──
setDT(pred_db)
setDT(joined_data)

# Initialize the output column
pred_db[, consolidated := NA_real_]

# ── Get the years that have a trained model ──
available_years <- names(rf_models_per_year)

# ── Set a chunk size to limit peak memory during predict() ──
CHUNK_SIZE <- 50000L # adjust based on available RAM

# ── Prediction loop ──
unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!(yr_char %in% available_years)) {
    cat(paste0("  Warning: No model found for year ", yr, "\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correctly subset to the current year (avoids the masking bug)
  test_idx <- which(joined_data$year == yr)
  pred_idx <- which(pred_db$year == yr)

  n <- length(test_idx)

  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, "\n"))
    next
  }

  # Predict in chunks to control peak memory
  preds <- numeric(n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    chunk <- joined_data[test_idx[s:e], ]
    preds[s:e] <- predict(rf_model, newdata = chunk)
  }

  # Fast update by reference
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)

  # Free memory after each year
  rm(chunk, preds)
  gc()
}

# ── Write output efficiently ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

- **Bug fix:** Replaced `filter(year == year)` with explicit `which(joined_data$year == yr)` using a distinctly named loop variable (`yr`), eliminating the self-comparison that caused every year to process the entire dataset.
- **Memory reduction:** Removed all unused library imports and the unused `st_read()` call; added `gc()` calls.
- **Chunked prediction:** Predictions are made in configurable chunks (`CHUNK_SIZE`) so that `predict()` never needs to allocate output for the full year at once.
- **`data.table` for speed:** Uses `set()` for in-place column updates (no copy-on-modify) and `fwrite()` for fast CSV output.
- **Models are preserved:** No retraining occurs; `rf_models_per_year` is used as-is.