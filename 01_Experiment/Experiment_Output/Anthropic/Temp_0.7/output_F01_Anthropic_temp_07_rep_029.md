 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always TRUE), meaning `test_set` contains the *entire* dataset every iteration, not just the rows for that year. This massively inflates computation.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Heavy Library Loading:** Many libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are loaded but not used in the prediction phase, consuming memory.

4. **Inefficient Row Indexing:** `pred_db$consolidated[pred_db$year == year]` performs a full-column logical scan on every iteration.

5. **`predict.randomForest` on Huge Data:** Even with the bug fixed, predicting hundreds of thousands of rows at once can spike memory, especially with large forests.

6. **Writing a Massive CSV:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Variable masking in `filter()` | Use `.env$year` or rename the loop variable |
| Unused `st_read` call | Remove it |
| Unnecessary libraries | Load only what is needed |
| Inefficient row assignment | Pre-split data by year using `split()` or `data.table` keying |
| Memory pressure from `predict()` | Predict in chunks within each year |
| Slow CSV write | Use `data.table::fwrite()` |
| General memory | Call `gc()` between years; convert to `data.table` |

---

## Optimized Working R Code

```r
# ── Load only required libraries ──────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models ───────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert prediction database and joined data to data.table ─────
setDT(pred_db)
setDT(joined_data)

# ── Initialise output column ─────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Key / index by year for fast subsetting ───────────────────────
setkey(pred_db, year)
setkey(joined_data, year)

# ── Chunk size for predict() to limit peak memory ────────────────
CHUNK_SIZE <- 50000L          # tune to your available RAM

# ── Available model years ─────────────────────────────────────────
model_years <- names(rf_models_per_year)

# ── Prediction loop ──────────────────────────────────────────────
for (yr in unique(pred_db$year)) {

  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!(yr_char %in% model_years)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ── Fast subset using data.table key (no full-table scan) ──────
  test_set <- joined_data[.(yr)]            # correct year filter
  n        <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s – skipping.\n", yr))
    next
  }

  # ── Predict in memory-friendly chunks ──────────────────────────
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (i in seq_along(starts)) {
    idx_start <- starts[i]
    idx_end   <- min(idx_start + CHUNK_SIZE - 1L, n)
    preds[idx_start:idx_end] <-
      predict(rf_model, newdata = test_set[idx_start:idx_end, ])
  }

  # ── Write predictions back (keyed join avoids logical scan) ────
  pred_db[.(yr), consolidated := preds]

  # ── Free memory between years ──────────────────────────────────
  rm(test_set, preds)
  gc()
}

# ── Fast write ────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

1. **Bug fix:** Replaced `filter(year == year)` with `joined_data[.(yr)]` — correctly subsets to the current year only.
2. **Removed** the unused `st_read()` call and all unnecessary library loads.
3. **Converted to `data.table`** with keyed subsetting for fast, memory-efficient year filtering.
4. **Chunked `predict()` calls** (configurable `CHUNK_SIZE`) so peak memory stays bounded.
5. **`gc()` between years** to release memory promptly.
6. **`fwrite()`** replaces `write.csv()` for significantly faster output.

No models are retrained; the existing `rf_models_per_year` object is used as-is.