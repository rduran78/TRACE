 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in this code:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always TRUE), so every iteration loads the *entire* dataset instead of a single year's subset. This is both a correctness bug and a massive performance drain.

2. **Monolithic `predict()` Call:** Calling `predict()` on hundreds of thousands of rows at once with a Random Forest model consumes enormous memory because the forest must allocate prediction arrays for every tree × every row simultaneously.

3. **Unnecessary Spatial Library Loading:** `st_read` loads the full spatial geometry of `prep_data`, but it is never used in the prediction loop. The loaded spatial object and unused libraries (`sf`, `spdep`, `terra`, `blockCV`, etc.) consume memory.

4. **Inefficient Data Structures:** Using a `tibble`/`data.frame` with `dplyr::filter` inside a loop is slower than keyed `data.table` operations for repeated subsetting.

5. **Full CSV Write at End:** Writing a very large CSV in one shot can spike memory. Streaming or chunked writes are safer.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable masking bug | Rename loop variable to `yr` and filter with `.env$yr` or use `data.table` keyed subset |
| Memory spike from large `predict()` | Predict in row-chunks (e.g., 50 000 rows at a time) |
| Unused spatial data & libraries | Don't load `prep_data` or unused libraries; load only what's needed |
| Slow subsetting | Convert `joined_data` and `pred_db` to `data.table` with key on `year` |
| Large CSV write | Use `data.table::fwrite` (faster, lower memory) |

The trained Random Forest models are **preserved exactly as-is**; no retraining occurs.

---

## Optimized Working R Code

```r
# ── Load only the libraries actually needed for prediction ──
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load pre-trained models ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This loads: rf_models_per_year, joined_data, pred_db  (adjust if names differ)

# ── Convert to data.table for fast keyed operations ──
setDT(joined_data)
setDT(pred_db)
setkey(joined_data, year)
setkey(pred_db, year)

# ── Initialise output column ──
pred_db[, consolidated := NA_real_]

# ── Chunked prediction function ──
#    Splits newdata into chunks to cap peak memory usage.
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0L))

  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Main prediction loop ──
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # Keyed subset – much faster than dplyr::filter inside a loop
  test_set <- joined_data[.(yr)]

  if (nrow(test_set) == 0L) {
    cat(sprintf("  Warning: No rows in joined_data for year %s.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Chunked predict to avoid memory spike
  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Write predictions back (keyed join)
  pred_db[.(yr), consolidated := preds]

  cat(sprintf("  Done – %d predictions written.\n", length(preds)))
}

# ── Write output efficiently ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("All predictions complete. File written: RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Bug fix:** Renamed loop variable to `yr` so the year filter actually selects the correct subset.
- **Chunked prediction:** `predict_chunked()` processes rows in batches of 50 000 (tunable), capping RAM usage per call.
- **`data.table` throughout:** Keyed subsetting (`[.(yr)]`) and `fwrite` replace slow `dplyr::filter` and `write.csv`.
- **Removed unused libraries and data loads:** `prep_data`, `sf`, `spdep`, `terra`, `blockCV`, `zoo`, `LongituRF`, `plm`, `fixest`, `scales`, `stringi`, and `tidyverse` are not loaded, freeing substantial memory.
- **Models untouched:** All `rf_models_per_year` objects are loaded and used as-is with no retraining.