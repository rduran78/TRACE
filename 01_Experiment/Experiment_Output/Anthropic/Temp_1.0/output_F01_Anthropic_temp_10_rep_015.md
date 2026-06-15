 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (because the loop variable and the column share the same name), returning *all* rows instead of just the target year. This means every iteration predicts on the entire dataset — massively inflating computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used, consuming significant memory (especially for hundreds of thousands of geometries).

3. **Full Data Frame in Memory:** `joined_data` and `pred_db` may be very large `data.frame` or `sf` objects. Subsetting with `dplyr::filter` on a large `sf` object is slower than on a plain `data.table`.

4. **`predict.randomForest` on Huge Batches:** Predicting hundreds of thousands of rows at once through a Random Forest can spike memory because the method internally allocates matrices across all trees.

5. **`write.csv` on a Large Object:** `write.csv` is significantly slower than alternatives like `fwrite()` from `data.table`.

6. **Library Bloat:** Many loaded libraries (`blockCV`, `zoo`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`) appear unused, each consuming memory at load time.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Unused `st_read` call | Remove it entirely |
| Unused libraries | Remove them to reduce memory footprint |
| Large data frame subsetting | Convert to `data.table` and use keyed subsetting |
| Memory spike during `predict()` | Chunk predictions into batches (e.g., 50 000 rows) |
| Slow CSV write | Use `data.table::fwrite()` |
| Garbage collection | Call `gc()` between years to free memory |

---

## Optimized Working R Code

```r
# ── Load only required libraries ──────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models ───────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert working data to data.table for speed ─────────────────
# (joined_data and pred_db are expected to exist after loading the .RData)
setDT(joined_data)
setDT(pred_db)

# ── Pre-allocate the output column ────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Key both tables on 'year' for fast subsetting ─────────────────
setkey(joined_data, year)
setkey(pred_db, year)

# ── Chunked predict helper (controls peak memory) ────────────────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))

  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ], predict.all = FALSE)
  }
  preds
}

# ── Main prediction loop ─────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }

  # Fast keyed subset (no copy overhead with data.table)
  test_set <- joined_data[.(yr)]

  rf_model <- rf_models_per_year[[yr_char]]

  # Chunked prediction to limit memory spikes
  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Assign predictions back
  pred_db[.(yr), consolidated := preds]

  # Free transient memory
  rm(test_set, preds)
  gc(verbose = FALSE)

  cat(sprintf("  Done — %d cells predicted.\n", length(preds)))
}

# ── Fast write ────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Bug fix:** Renamed loop variable to `yr` so `joined_data[.(yr)]` correctly filters only the target year's rows.
- **Removed** the unused `st_read()` call and all unnecessary `library()` calls — freeing potentially gigabytes of memory.
- **Converted** to `data.table` with keyed subsetting — orders of magnitude faster than `dplyr::filter` on large frames.
- **Chunked `predict()`** in batches of 50 000 rows to cap peak memory usage per call.
- **Replaced `write.csv`** with `fwrite()` for much faster output.
- **Added `gc()`** after each year to return freed memory to the OS promptly.
- **No models were retrained** — the existing `rf_models_per_year` list is used as-is.