 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object (`prep_data`) for no apparent reason:** `st_read` reads a potentially massive shapefile into memory but `prep_data` is never used in the prediction step.

3. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` always equals the entire `joined_data` dataframe. This means every iteration predicts on *all* rows, not just the target year, wasting enormous compute and producing incorrect results.

4. **Predicting all at once per year:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows in a single call with a large Random Forest can spike memory because `randomForest::predict` builds a full matrix of responses across all trees.

5. **Using a `data.frame` for large row-assignment:** Assigning predictions back into a large `data.frame` column via logical indexing inside a loop is inefficient. `data.table` would be faster.

6. **Writing a massive CSV at the end:** `write.csv` is slow for large files; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and objects | Remove `st_read` call and unused `library()` calls to free memory |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) or use `.env$year` with `dplyr` |
| Whole-dataset prediction per iteration | After fixing the filter, predict only the relevant year-subset |
| Memory spike during `predict()` | Chunk large year-subsets into batches (e.g., 50 000 rows) |
| Slow data.frame assignment | Convert `pred_db` to `data.table` and assign by reference |
| Slow CSV writing | Use `data.table::fwrite` |
| General memory | Call `gc()` after each year; optionally remove `joined_data` columns not used as predictors |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(randomForest)
library(tidyverse)
library(data.table)

# ── Load pre-trained models and associated data ───────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This is expected to provide: rf_models_per_year, pred_db, joined_data

# ── Convert to data.table for fast by-reference operations ────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Chunked predict helper (controls peak memory) ────────────────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# ── Main prediction loop ─────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))
  
  yr_char <- as.character(yr)
  
  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " – skipping.\n"))
    next
  }
  
  # Correctly subset the data for this year only (avoids the shadowing bug)
  test_idx <- which(joined_data$year == yr)
  test_set <- joined_data[test_idx]
  
  rf_model <- rf_models_per_year[[yr_char]]
  
  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)
  
  # Assign by reference into pred_db
  pred_db[year == yr, consolidated := preds]
  
  # Free memory
  rm(test_set, preds)
  gc()
  
  cat(paste0("  Done – ", length(test_idx), " cells predicted.\n"))
}

# ── Write results efficiently ─────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Removed** the unused `st_read()` call and nine unnecessary library imports, reclaiming significant memory.
- **Fixed the critical filter bug** by renaming the loop variable to `yr`, so each iteration now correctly subsets only the rows for that year.
- **Converted data frames to `data.table`** for fast by-reference column updates (`:=`) and fast subsetting.
- **Added chunked prediction** (`predict_chunked`) to cap peak memory: predictions are made in batches of 50 000 rows instead of all at once.
- **Added `gc()` after each year** to release memory promptly.
- **Replaced `write.csv` with `fwrite`** for dramatically faster file output.
- **All pre-trained Random Forest models are preserved untouched**—no retraining occurs.