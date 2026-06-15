 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`st_read`) that is never used in prediction:** `prep_data` is read but plays no role in the loop. This wastes significant memory, especially for hundreds of thousands of geometries.

3. **The `.RData` file likely loads the entire training environment:** This may include `joined_data`, `pred_db`, the trained models, and possibly the full training data — all held in memory simultaneously.

4. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly because the column name `year` and the loop variable `year` are identical. This causes `dplyr::filter` to evaluate `year == year` as `TRUE` for every row, meaning **the entire dataset is passed to `predict()` every iteration**, not just one year's worth. This is the single biggest performance problem.

5. **`pred_db` may be a large data.frame:** Assigning into it row-by-row-group with base R indexing (`pred_db$consolidated[pred_db$year == year]`) is acceptable but the real cost is the `predict()` call receiving the full dataset each time due to the bug above.

6. **`randomForest::predict` on hundreds of thousands of rows is memory-intensive:** The predict method internally allocates matrices proportional to `n_rows × n_trees × n_nodes`.

7. **Writing a potentially huge CSV with `write.csv`:** This is slow for large data; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| # | Action | Rationale |
|---|--------|-----------|
| 1 | Remove unused libraries and the unused `prep_data` read | Free memory |
| 2 | Fix the variable-masking bug (`year == year`) | Correct filtering → dramatically fewer rows per `predict()` call |
| 3 | Use `data.table` for `pred_db` and `joined_data` | Faster subsetting and assignment |
| 4 | Optionally chunk large year-groups for `predict()` | Caps peak memory |
| 5 | Use `data.table::fwrite` instead of `write.csv` | Much faster I/O |
| 6 | Call `gc()` between years | Return memory to OS between large allocations |
| 7 | Only load needed objects from `.RData` | Avoid loading training data if it is bundled in the file |

---

## Optimized R Code

```r
# ── 1. Load only the libraries actually needed ──────────────────────────
library(data.table)
library(randomForest)   # for predict.randomForest
library(dplyr)          # only if joined_data must stay as tibble

# ── 2. Load models and prediction data ──────────────────────────────────
#    If the .RData contains objects you do NOT need (e.g. training data),
#    load into a temporary environment and extract only what is required.
tmp_env <- new.env()
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData',
     envir = tmp_env)

rf_models_per_year <- tmp_env$rf_models_per_year
joined_data        <- as.data.table(tmp_env$joined_data)
pred_db            <- as.data.table(tmp_env$pred_db)

rm(tmp_env)
gc()

# ── 3. Initialise the output column ─────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── 4. Key the tables for fast subsetting ────────────────────────────────
setkey(joined_data, year)
setkey(pred_db, year)

# ── 5. Chunked prediction loop (fixed masking bug) ──────────────────────
CHUNK_SIZE <- 50000L   # rows per chunk; tune to your available RAM

years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct subset: use the loop variable 'yr', not 'year'
  test_set <- joined_data[year == yr]

  n <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s – skipping.\n", yr))
    next
  }

  # Predict in chunks to cap peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # Assign back (data.table keyed join – very fast)
  pred_db[year == yr, consolidated := preds]

  # Free memory before next year
  rm(test_set, preds)
  gc()
}

# ── 6. Write output efficiently ──────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Expected Impact

| Issue | Before | After |
|-------|--------|-------|
| Filtering bug (`year == year`) | Entire dataset predicted every iteration | Only one year's rows predicted |
| Unused shapefile in memory | Hundreds of MB wasted | Not loaded |
| Unused libraries | ~15 packages loaded | 3 packages loaded |
| Peak memory during `predict()` | Unbounded (full data × trees) | Capped by `CHUNK_SIZE` |
| CSV write speed | `write.csv` (slow) | `fwrite` (10–50× faster) |
| Subsetting speed | `dplyr::filter` on data.frame | `data.table` keyed lookup |

The **single most impactful fix** is correcting the variable-masking bug (item 2 / loop variable renamed to `yr`). This alone can reduce per-iteration work by a factor equal to the number of unique years. Combined with the memory optimizations, the script should run comfortably on a standard personal computer.