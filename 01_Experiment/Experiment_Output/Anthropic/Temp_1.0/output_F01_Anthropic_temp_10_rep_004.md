 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (not to the loop variable), returning *all* rows every iteration instead of just the rows for that year. This means the Random Forest `predict()` call processes the entire dataset each iteration — massively multiplying the workload.

2. **Loading Unnecessary Spatial Data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Inefficient Data Structures:** Using `tidyverse` (tibble/data.frame) operations for row subsetting and assignment on hundreds of thousands of rows per year is slower than `data.table` keyed operations.

4. **Monolithic Predict Calls:** Calling `predict()` on a very large `newdata` in one shot can spike memory, especially with large Random Forest ensembles (each tree votes on every row simultaneously).

5. **Excessive Library Loading:** Many libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `scales`, `spdep`, etc.) are loaded but unused, consuming memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Unused libraries | Remove all unnecessary `library()` calls |
| Slow subsetting/assignment | Convert to `data.table` with keying |
| Memory spike on large predict | Predict in chunks (batches) |
| Output format | Use `fwrite()` instead of `write.csv()` |

---

## Optimized Working R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load pre-trained models and prediction database ───────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expects: rf_models_per_year (named list), joined_data, pred_db

# ── Convert to data.table for speed ──────────────────────────────
setDT(pred_db)
setDT(joined_data)

# Key both tables on year for fast subsetting
setkey(pred_db,    year)
setkey(joined_data, year)

# Initialise consolidated column
pred_db[, consolidated := NA_real_]

# ── Chunked prediction function ──────────────────────────────────
# Predicts in batches to limit peak memory usage.
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n   <- nrow(newdata)
  out <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    out[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  out
}

# ── Main prediction loop ─────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # Fast keyed subset (no copy overhead with data.table)
  test_set <- joined_data[.(yr)]          # rows where year == yr
  rf_model <- rf_models_per_year[[yr_char]]

  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Fast keyed update-by-reference
  pred_db[.(yr), consolidated := preds]

  cat(sprintf("  Done – %d cells predicted.\n", length(preds)))
}

# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output saved to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Bug fix:** Replaced `filter(year == year)` with `data.table` keyed join `joined_data[.(yr)]`, which correctly subsets only the rows matching the current loop year.
2. **Memory reduction:** Removed the unused `st_read()` call and 11 unnecessary library imports, freeing potentially gigabytes of RAM.
3. **Speed improvement:** Switched from `dplyr` filtering and base-R indexed assignment to `data.table` keyed subsetting and update-by-reference (`:=`), which avoids copying the entire data frame.
4. **Chunked prediction:** `predict_chunked()` feeds rows to the Random Forest in batches of 50,000 (adjustable), capping peak memory during the vote-aggregation step.
5. **Faster I/O:** `fwrite()` replaces `write.csv()` for significantly faster CSV output.
6. **Models are preserved as-is** — no retraining occurs.