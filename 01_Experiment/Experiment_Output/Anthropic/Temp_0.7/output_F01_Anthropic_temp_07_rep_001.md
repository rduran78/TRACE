 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning *all* rows every iteration. This means the full dataset is predicted repeatedly for every unique year — a massive waste of computation and likely the primary cause of slowness.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes significant memory for no benefit.

3. **Monolithic `predict()` calls on huge data frames:** With hundreds of thousands of rows and many predictors, a single `predict()` call on a Random Forest can spike memory because every tree must be evaluated across all rows simultaneously.

4. **Using `tidyverse` filtering inside a loop:** `dplyr::filter()` on a large data frame inside a tight loop adds overhead compared to index-based or `data.table` subsetting.

5. **Writing the full data frame to CSV at the end:** For very large data, `write.csv()` is slow; `data.table::fwrite()` is substantially faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so the filter actually subsets by year. |
| Unused shapefile load | Remove the `st_read()` call entirely. |
| Memory-heavy predict | Predict in row-chunks (e.g., 50 000 rows at a time) to cap peak memory. |
| Slow subsetting | Convert working data to `data.table` and use keyed subsetting. |
| Slow CSV write | Use `data.table::fwrite()`. |
| Library bloat | Load only the packages actually needed for prediction. |

---

## Optimized R Code

```r
# ── Only load what is actually needed for prediction ──
library(data.table)
library(randomForest)

# ── Load the pre-trained models ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert both frames to data.table for fast ops ──
pred_dt   <- as.data.table(pred_db)
joined_dt <- as.data.table(joined_data)

# ── Pre-allocate the output column ──
pred_dt[, consolidated := NA_real_]

# ── Set keys for fast subsetting ──
setkey(pred_dt,   year)
setkey(joined_dt, year)

# ── Chunked prediction function (controls peak memory) ──
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── Main prediction loop ──
years <- unique(pred_dt$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))
  
  yr_char <- as.character(yr)
  
  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s — skipping.\n", yr))
    next
  }
  
  # Correct subsetting: 'yr' is the loop variable, not the column name
  test_chunk <- joined_dt[year == yr]
  
  if (nrow(test_chunk) == 0L) {
    cat(sprintf("  Warning: No test data for year %s — skipping.\n", yr))
    next
  }
  
  rf_model <- rf_models_per_year[[yr_char]]
  
  preds <- predict_chunked(rf_model, test_chunk, chunk_size = 50000L)
  
  pred_dt[year == yr, consolidated := preds]
  
  cat(sprintf("  Done — %d predictions written.\n", length(preds)))
}

# ── Fast write ──
fwrite(pred_dt, "RF_imputated_db.csv")

cat("All predictions complete. File written.\n")
```

---

## Summary of Changes

- **Bug fix:** The loop variable is now `yr`, so `joined_dt[year == yr]` correctly subsets to one year at a time instead of returning the entire dataset.
- **Removed unused `st_read()`** call — saves potentially gigabytes of RAM.
- **Chunked `predict()`** caps peak memory by processing 50 000 rows at a time (tunable via `chunk_size`).
- **`data.table`** keyed subsetting replaces `dplyr::filter()` for faster row selection.
- **`fwrite()`** replaces `write.csv()` for significantly faster output.
- **Reduced library loads** to only `data.table` and `randomForest`, lowering startup time and memory footprint.
- **No models are retrained** — only `predict()` is called on the existing `rf_models_per_year`.