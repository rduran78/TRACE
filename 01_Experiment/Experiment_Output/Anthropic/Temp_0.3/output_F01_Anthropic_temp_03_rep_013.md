 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (because the loop variable `year` shadows the column name), returning **all rows** instead of the intended subset. This means every iteration predicts on the entire dataset, massively inflating computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used, consuming memory for nothing.

3. **Full `joined_data` in Memory:** The entire `joined_data` object (potentially millions of rows with many columns) is kept in memory and filtered repeatedly inside the loop.

4. **`predict()` on Huge Data Frames:** `randomForest::predict()` on hundreds of thousands of rows with many predictors is memory-intensive, especially if the forest is large (many trees, deep nodes).

5. **Writing a Massive CSV at the End:** `write.csv()` on a very large data frame is slow and produces a large file.

6. **No Garbage Collection:** Large intermediate objects are never removed.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Use a distinctly named loop variable (e.g., `yr`) |
| Unused `prep_data` load | Remove it |
| Entire `joined_data` in memory during predict | Split prediction into chunks within each year |
| Large prediction vectors | Pre-allocate and assign by index |
| Slow CSV write | Use `data.table::fwrite()` |
| Memory pressure | Use `gc()` after each year; keep only needed columns for prediction |
| Potential further speedup | Optionally use `ranger` for prediction only (not applicable here since models are `randomForest` objects — but chunking helps) |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)
library(tidyverse)

# ── 1. Load only what is needed ──────────────────────────────────────────────
# Do NOT load prep_data — it is unused and wastes memory.
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Convert pred_db and joined_data to data.tables for speed ──────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── 3. Identify predictor columns once ───────────────────────────────────────
# Extract the variable names the RF models expect from any available model.
sample_model <- rf_models_per_year[[1]]
predictor_cols <- rownames(sample_model$importance)

# ── 4. Set a chunk size that fits comfortably in RAM ─────────────────────────
CHUNK_SIZE <- 50000L   # adjust up or down based on available RAM

# ── 5. Loop over years with correct variable naming ──────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Subset only the rows and columns needed for prediction
  idx <- which(joined_data$year == yr)
  test_subset <- joined_data[idx, ..predictor_cols]   # only predictor columns

  n <- nrow(test_subset)
  preds <- numeric(n)

  # Predict in chunks to limit peak memory
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_subset[s:e, ])
  }

  # Write predictions back into pred_db
  pred_db[year == yr, consolidated := preds]

  # Free memory

  rm(test_subset, preds, rf_model)
  gc()

  cat(paste0("  Done — predicted ", n, " rows.\n"))
}

# ── 6. Write output efficiently ──────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("All predictions written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

| # | Change | Why |
|---|---|---|
| 1 | Renamed loop variable to `yr` | **Fixes the critical bug** where `filter(year == year)` returned all rows |
| 2 | Removed `st_read()` of unused shapefile | Saves potentially hundreds of MB of RAM |
| 3 | Switched to `data.table` (`setDT`, `..predictor_cols`, `fwrite`) | Faster subsetting, column selection, and file writing |
| 4 | Select only predictor columns for `predict()` | Avoids copying irrelevant columns into the prediction call |
| 5 | Chunked `predict()` in batches of 50 000 rows | Caps peak memory; prevents R from allocating one enormous matrix |
| 6 | `rm()` + `gc()` after each year | Returns memory to the OS promptly |
| 7 | `fwrite()` instead of `write.csv()` | Typically 5–10× faster for large files |
| 8 | **No models are retrained** | Only `predict()` is called on existing `rf_models_per_year` |