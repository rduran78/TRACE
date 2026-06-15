 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop, consuming memory for nothing.
2. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column to itself (the loop variable `year` is masked by the column name `year`), so `test_set` is always the *entire* `joined_data` frame rather than a single-year subset. This means every iteration predicts on hundreds of thousands of rows unnecessarily and writes incorrect results.
3. **Monolithic `predict()` on a huge data frame:** Even after fixing the filter, calling `predict()` on a very large `test_set` in one shot can spike memory because `randomForest::predict` builds a full matrix of tree-level predictions internally.
4. **Using a `data.frame` for row-level assignment:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a large data frame is slow; `data.table` set-by-reference is far faster.
5. **Unnecessary libraries loaded:** Many libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `zoo`, `terra`) are loaded but unused, each consuming memory and load time.
6. **Writing a massive CSV at the end:** `write.csv` on a large frame is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused `st_read` of large shapefile | Remove it |
| Unused library loading | Remove unnecessary `library()` calls |
| Variable shadowing in `filter()` | Use `.env$year` or a renamed loop variable |
| Whole-dataset predict per iteration | Fix filter so only one year's rows are predicted |
| Memory spike in `predict()` | Chunk large year-subsets into batches |
| Slow row assignment in data.frame | Convert to `data.table` and use `:=` by reference |
| Slow `write.csv` | Use `fwrite()` |
| Optional: parallelism | Not needed once the above are fixed, but `predict` can be chunked in parallel if desired |

---

## Optimized R Code

```r
# ── Only load what is actually needed ──────────────────────────────
library(data.table)
library(randomForest)
library(tidyverse)        # kept only if joined_data is a tibble

# ── Load pre-trained models and associated data ────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, joined_data, pred_db

# ── Convert working tables to data.table for speed ─────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Chunked predict helper (controls peak memory) ──────────────────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  # Pre-allocate result vector
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# ── Main prediction loop ──────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # Correct subset: use the renamed loop variable 'yr'
  test_idx  <- which(joined_data$year == yr)
  test_set  <- joined_data[test_idx]

  rf_model  <- rf_models_per_year[[yr_char]]

  preds     <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Fast set-by-reference assignment
  pred_rows <- which(pred_db$year == yr)
  set(pred_db, i = pred_rows, j = "consolidated", value = preds)

  # Free memory eagerly

  rm(test_set, preds, rf_model)
  gc()
}

# ── Write output efficiently ───────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

1. **Removed** the unused `st_read()` call and all unnecessary library imports → **immediate memory savings**.
2. **Fixed the filtering bug** by renaming the loop variable to `yr` so that `joined_data$year == yr` correctly subsets a single year.
3. **Converted to `data.table`** and used `set()` for by-reference column updates → **avoids repeated deep copies**.
4. **Added `predict_chunked()`** to break large prediction calls into configurable batches (default 50 000 rows) → **caps peak memory** inside `randomForest::predict`.
5. **Explicit `rm()` + `gc()`** after each year frees intermediate objects promptly.
6. **Replaced `write.csv` with `fwrite()`** → typically 5–10× faster on large files.

No models are retrained; all existing `rf_models_per_year` objects are used as-is.