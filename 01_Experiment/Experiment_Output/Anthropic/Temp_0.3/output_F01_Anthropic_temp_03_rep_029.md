 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` always equals the entire `joined_data` dataframe. This means every iteration predicts on *all* rows, not just the rows for that year — massively inflating computation time and producing incorrect results.

3. **Predicting all rows at once per iteration:** Even after fixing the bug, calling `predict()` on hundreds of thousands of rows in a single call with a large Random Forest model can spike memory usage because `randomForest::predict` builds a full matrix of predictions across all trees internally.

4. **Using a base `data.frame` for column assignment in a loop:** Repeated assignment to `pred_db$consolidated[pred_db$year == year]` on a very large data.frame is slow due to repeated logical indexing and potential memory copies.

5. **Writing a massive CSV at the end:** `write.csv()` on a very large data.frame is slow; `data.table::fwrite()` is significantly faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read()` call | Remove it entirely to free memory. |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` / `!!year` to disambiguate. |
| Memory spike from large single `predict()` calls | Predict in chunks (batches) within each year to cap peak memory. |
| Slow indexing on `data.frame` | Convert `pred_db` to a `data.table` and use keyed joins or indexed updates. |
| Slow CSV write | Use `data.table::fwrite()`. |
| General | Call `gc()` between years; remove objects no longer needed. |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)
library(tidyverse)

# ── 1. Load only what is needed ──────────────────────────────────────────────
# Do NOT load the shapefile — it is unused and wastes memory.
# load prep_data only if truly needed downstream.

load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Convert to data.table for fast indexed operations ─────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── 3. Set a chunk size that fits comfortably in RAM ─────────────────────────
CHUNK_SIZE <- 50000L   # tune this: smaller = less RAM, slightly slower

# ── 4. Prediction loop — fixed variable shadowing, chunked predict ───────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ---- Get the row indices for this year in both tables ----
  test_idx   <- which(joined_data$year == yr)
  pred_idx   <- which(pred_db$year == yr)

  n <- length(test_idx)
  cat(paste0("  Rows to predict: ", n, "\n"))

  # ---- Predict in chunks to limit peak memory ----
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)

    chunk_test <- joined_data[test_idx[s:e], ]
    preds      <- predict(rf_model, newdata = chunk_test)

    set(pred_db, i = pred_idx[s:e], j = "consolidated", value = preds)

    # Free chunk memory immediately
    rm(chunk_test, preds)
  }

  # Free the model copy and force garbage collection
  rm(rf_model)
  gc(verbose = FALSE)

  cat(paste0("  Done with year ", yr, ".\n"))
}

# ── 5. Write output efficiently ──────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("All predictions written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | What changed | Why |
|---|---|---|
| 1 | Removed `st_read()` of unused shapefile | Saves potentially gigabytes of RAM. |
| 2 | Renamed loop variable to `yr` | Fixes the critical **variable-shadowing bug** that caused every year to predict on the full dataset. |
| 3 | Converted to `data.table` + used `set()` | Avoids repeated deep copies; updates cells in-place. |
| 4 | Chunked `predict()` calls | Caps peak memory; each chunk processes ≤ 50 000 rows. |
| 5 | Added `rm()` + `gc()` per iteration | Releases memory between years. |
| 6 | Replaced `write.csv()` with `fwrite()` | Typically 10–50× faster for large files. |
| 7 | **No models were retrained** | All existing `rf_models_per_year` objects are used as-is. |