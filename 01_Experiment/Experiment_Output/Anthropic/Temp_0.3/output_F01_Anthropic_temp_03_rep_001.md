 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The script has several performance and correctness issues:

1. **Critical Bug — Variable Masking in `filter()`:** `filter(year == year)` compares the column `year` to itself (always `TRUE`), so `test_set` contains the *entire* dataset every iteration. This means every year's prediction runs on all rows, massively inflating computation time and producing incorrect results.

2. **Loading Unnecessary Spatial Data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Entire Data Frame Held as `sf`/`tibble`:** If `joined_data` is an `sf` object, the geometry column consumes significant memory and is not needed for `predict()`.

4. **Row-by-Row Assignment via Logical Indexing on a Large Data Frame:** `pred_db$consolidated[pred_db$year == year] <- ...` forces a full-column scan each iteration.

5. **All Predictor Columns Loaded at Once:** If `joined_data` has many unused columns, they waste memory during `predict()`.

6. **No Garbage Collection or Chunking:** For hundreds of thousands of rows, calling `predict()` on the full year subset in one shot can spike memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable masking bug | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Geometry overhead | Drop geometry with `st_drop_geometry()` before the loop |
| Unnecessary columns | Select only the columns the RF model expects, plus `year` |
| Large single-pass predict | Chunk predictions within each year if memory is tight |
| Slow CSV write | Use `data.table::fwrite()` |
| Assignment efficiency | Pre-split data by year using `split()` or `data.table` keying |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)
library(sf)

# ── 1. Load only what is needed ──────────────────────────────────────────────
# Do NOT load the shapefile — it is unused in prediction.
# prep_data <- st_read(...)   # REMOVED

load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Drop geometry if joined_data is an sf object ─────────────────────────
if (inherits(joined_data, "sf")) {
  joined_data <- st_drop_geometry(joined_data)
}

# ── 3. Convert to data.table for speed ──────────────────────────────────────
setDT(joined_data)
setDT(pred_db)

# ── 4. Identify required predictor columns from one of the models ───────────
sample_model <- rf_models_per_year[[1]]
predictor_cols <- rownames(importance(sample_model))

# Keep only year + predictors in joined_data to free memory
keep_cols <- intersect(names(joined_data), c("year", predictor_cols))
joined_data <- joined_data[, ..keep_cols]
gc()

# ── 5. Initialise the output column ─────────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── 6. Predict year-by-year (with optional chunking) ────────────────────────
available_years <- names(rf_models_per_year)
CHUNK_SIZE <- 50000L          # tune to your RAM; set to Inf to disable chunking

for (yr in unique(pred_db$year)) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% available_years) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # FIX: use yr (loop variable), not the column name

  test_set <- joined_data[year == yr, ..predictor_cols]

  n <- nrow(test_set)
  preds <- numeric(n)

  # Chunk predictions to cap peak memory
  starts <- seq(1L, n, by = CHUNK_SIZE)
  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  pred_db[year == yr, consolidated := preds]

  # Free memory after each year
  rm(test_set, preds)
  gc()
}

# ── 7. Write output efficiently ─────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Fixed `filter(year == year)` → `joined_data[year == yr]` | **Correctness** — each year now predicts on the right subset only |
| 2 | Removed unused `st_read()` | Saves potentially gigabytes of RAM |
| 3 | `st_drop_geometry()` | Drops heavy geometry column |
| 4 | Subset to predictor columns only | Reduces memory footprint substantially |
| 5 | Chunked `predict()` calls | Caps peak memory during prediction |
| 6 | `data.table` instead of `tibble`/`dplyr` | Faster subsetting and assignment |
| 7 | `fwrite()` instead of `write.csv()` | Orders-of-magnitude faster file writing |
| 8 | `gc()` after each year | Returns freed memory to the OS promptly |

No models are retrained; only the prediction loop and data handling are optimized.