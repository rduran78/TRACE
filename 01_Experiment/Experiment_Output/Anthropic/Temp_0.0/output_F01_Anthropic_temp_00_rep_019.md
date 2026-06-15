 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile via `st_read()` but is never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` becomes the **entire dataset** every iteration — massively inflating computation and producing incorrect results.

4. **Full data frame held in memory:** `joined_data` and `pred_db` may be very large `data.frame` or `tibble` objects. Predicting on the entire (incorrectly filtered) dataset repeatedly is extremely wasteful.

5. **`predict.randomForest` on huge data:** Even with correct filtering, calling `predict()` on hundreds of thousands of rows with many predictors in a single call can spike memory, especially if the forest is large (many trees, deep nodes).

6. **Writing a massive CSV at the end:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| Unused `prep_data` shapefile | Remove the `st_read()` call entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Monolithic prediction | Split prediction into chunks within each year to cap memory usage |
| Slow CSV write | Use `data.table::fwrite()` |
| General memory | Call `gc()` after each year; convert to `data.table` for efficiency |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load pre-trained models and data ──────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for speed and lower memory overhead
if (!is.data.table(pred_db))    setDT(pred_db)
if (!is.data.table(joined_data)) setDT(joined_data)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# ── Configurable chunk size (tune to your RAM) ───────────────────────────
CHUNK_SIZE <- 50000L   # rows per predict() call; lower if memory is tight

# ── Prediction loop ──────────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct filter: use the loop variable, not the column vs itself
  idx_joined <- which(joined_data$year == yr)
  idx_pred   <- which(pred_db$year == yr)

  n <- length(idx_joined)
  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, ".\n"))
    next
  }

  # Predict in chunks to cap peak memory
  n_chunks <- ceiling(n / CHUNK_SIZE)
  preds <- vector("numeric", n)

  for (ch in seq_len(n_chunks)) {
    start <- (ch - 1L) * CHUNK_SIZE + 1L
    end   <- min(ch * CHUNK_SIZE, n)
    chunk_rows <- idx_joined[start:end]
    preds[start:end] <- predict(rf_model, newdata = joined_data[chunk_rows, ])
  }

  # Write predictions back
  pred_db[idx_pred, consolidated := preds]

  # Free memory before next year
  rm(rf_model, preds)
  gc()

  cat(paste0("  Done — ", n, " rows predicted.\n"))
}

# ── Write output (fwrite is ~5-10x faster than write.csv) ────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Removed** all unused `library()` calls and the unused `st_read()` — this alone can free gigabytes of RAM.
- **Fixed the filtering bug** (`year == year` → `joined_data$year == yr`) so each year only processes its own rows.
- **Chunked prediction** via `CHUNK_SIZE` keeps peak memory bounded; adjust the value downward if you still hit limits.
- **Switched to `data.table`** for faster subsetting and in-place column assignment (`:=`).
- **Used `fwrite()`** instead of `write.csv()` for significantly faster output.
- **Explicit `gc()`** after each year releases memory from the prior model and predictions.
- **Pre-trained models are untouched** — no retraining occurs.