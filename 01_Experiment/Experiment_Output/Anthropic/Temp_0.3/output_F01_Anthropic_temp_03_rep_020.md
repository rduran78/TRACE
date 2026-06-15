 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (because the loop variable and the column share the same name), so `test_set` contains **all rows** every iteration instead of just one year's worth. This means the full dataset is predicted repeatedly for every unique year — massively multiplying computation time.

3. **Predicting on the entire dataset at once:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows in a single call with a large Random Forest model can spike memory usage because `randomForest::predict` must pass every row through every tree.

4. **Using a data.frame for large row-assignment:** Assigning predictions back into a column of a large `data.frame` (`pred_db$consolidated[pred_db$year == year]`) with repeated logical indexing is slow. `data.table` would be far more efficient.

5. **Writing a massive CSV:** `write.csv` on a very large data.frame is slow; `data.table::fwrite` is significantly faster.

6. **All libraries loaded upfront:** Several libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `zoo`, `terra`) are not needed for prediction and consume memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read` call | Remove it |
| Unused library loads | Remove them to reduce memory footprint |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` |
| Entire dataset predicted every iteration | Fix the filter so only one year's rows are predicted |
| Memory spike from large single `predict()` call | Predict in chunks (batches) within each year |
| Slow row-assignment on data.frame | Convert `pred_db` to `data.table` and use keyed assignment |
| Slow `write.csv` | Use `data.table::fwrite` |

---

## Optimized R Code

```r
# ── Only load what is actually needed for prediction ──
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data is a tibble / uses dplyr verbs

# ── Load the saved models and associated data ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for fast grouped operations ──
pred_dt    <- as.data.table(pred_db)
joined_dt  <- as.data.table(joined_data)

pred_dt[, consolidated := NA_real_]

# ── Set a batch size that fits comfortably in RAM ──
BATCH_SIZE <- 50000L   # tune up or down depending on available memory

# ── Prediction loop — one year at a time, in chunks ──
years <- unique(pred_dt$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── Correctly subset to the current year ──
  test_rows <- joined_dt[year == yr]
  n         <- nrow(test_rows)

  if (n == 0L) {
    cat(paste0("  No test rows for year ", yr, " — skipping.\n"))
    next
  }

  # ── Predict in batches to limit peak memory ──
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = BATCH_SIZE)

  for (s in starts) {
    e <- min(s + BATCH_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_rows[s:e, ])
  }

  # ── Write predictions back via fast data.table assignment ──
  pred_dt[year == yr, consolidated := preds]

  # ── Free memory after each year ──
  rm(test_rows, preds, rf_model)
  gc()
}

# ── Fast CSV write ──
fwrite(pred_dt, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

- **Removed** the unused `st_read()` call and all unnecessary `library()` calls → immediate memory savings.
- **Fixed the variable-shadowing bug** (`year == year` → loop variable renamed to `yr`) → each year now processes only its own rows instead of the entire dataset.
- **Converted to `data.table`** → faster subsetting and column assignment.
- **Chunked `predict()` calls** via `BATCH_SIZE` → caps peak memory so the process can run on a normal machine. Adjust `BATCH_SIZE` up (faster) or down (less RAM) as needed.
- **Explicit `rm()` + `gc()`** after each year → reclaims memory between iterations.
- **Replaced `write.csv` with `fwrite`** → dramatically faster file output.
- **No models are retrained** — only `predict()` is called on the existing `rf_models_per_year`.