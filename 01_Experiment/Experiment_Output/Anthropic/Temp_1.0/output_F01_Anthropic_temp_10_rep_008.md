 # Diagnosis and Optimization Strategy for Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

### 1. Critical Bug: Variable Masking in `filter()`
```r
test_set <- joined_data %>% filter(year == year)
```
This compares the column `year` to **itself** (always `TRUE`), so `test_set` contains the **entire dataset** every iteration, not just the target year. This massively inflates prediction time and memory usage.

### 2. Memory Overload from Unnecessary Spatial Data
`st_read()` loads a full spatial object (`prep_data`) with geometry, but it is never actually used in the prediction loop. Loading large shapefiles consumes significant RAM unnecessarily.

### 3. `predict()` on Huge Data Frames
With hundreds of thousands of rows (incorrectly unfiltered, making it worse), calling `predict()` on a `randomForest` object processes every tree for every row in memory. Even when the filter bug is fixed, the per-year slices can still be very large.

### 4. Row-by-Row Assignment into a Data Frame
```r
pred_db$consolidated[pred_db$year == year] <- predict(...)
```
Repeated subsetting and assignment on a large data frame inside a loop is slow in base R.

### 5. Writing a Massive CSV
`write.csv()` on a data frame with hundreds of thousands (or millions) of rows is slow and produces a very large file.

### 6. Loading Unnecessary Libraries
Many loaded libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, etc.) are not used in the prediction step and consume memory on load.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Filter bug | Use `!!year` or rename loop variable to `.year` to avoid masking |
| Unused spatial data | Remove `st_read()` call entirely |
| Memory pressure | Convert to `data.table`; process and predict in chunks if needed |
| Slow CSV write | Use `data.table::fwrite()` |
| Unused libraries | Remove them to reduce memory footprint |
| Large predictions | Optionally predict in row-chunks within each year to cap peak memory |

---

## Optimized Working R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load pre-trained models ──────────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expects: rf_models_per_year (named list), joined_data, pred_db

# ── Convert to data.table for speed ─────────────────────────────────
setDT(pred_db)
setDT(joined_data)

# ── Pre-allocate the output column ───────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Define a safe chunk-predict function to limit peak memory ────────
#    Adjust chunk_size downward if RAM is still tight.
chunk_predict <- function(model, newdata, chunk_size = 50000L) {

  n <- nrow(newdata)
  if (n == 0L) return(numeric(0L))
  
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
}

# ── Prediction loop (fixed filter bug) ──────────────────────────────
years <- unique(pred_db$year)

for (.yr in years) {
  yr_char <- as.character(.yr)
  cat("Predicting for year", yr_char, "...\n")
  
  if (!yr_char %in% names(rf_models_per_year)) {
    cat("  Warning: No model found for year", yr_char, "– skipping.\n")
    next
  }
  
  rf_model <- rf_models_per_year[[yr_char]]
  
  # ── Correct filter: use .yr, not the column name ──────────────────
  test_idx   <- which(joined_data$year == .yr)
  test_slice <- joined_data[test_idx]
  
  pred_idx <- which(pred_db$year == .yr)
  
  cat("  Rows to predict:", length(pred_idx), "\n")
  
  # ── Chunk-wise prediction to limit peak memory ────────────────────
  pred_db[pred_idx, consolidated := chunk_predict(rf_model, test_slice)]
  
  # ── Free memory immediately ────────────────────────────────────────
  rm(test_slice, rf_model)
  gc()
}

# ── Fast write ───────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

1. **Fixed the critical `filter(year == year)` bug** by renaming the loop variable to `.yr` and using base subsetting on `data.table`, ensuring only the correct year's rows are predicted.
2. **Removed `st_read()`** and all unused library loads to free hundreds of MB of RAM.
3. **Converted `pred_db` and `joined_data` to `data.table`** for fast subsetting and in-place column updates (no copy-on-modify).
4. **Added chunk-wise prediction** (`chunk_predict()`) so that `predict()` never needs to hold all rows × all trees in memory simultaneously. The `chunk_size` parameter (default 50,000) can be tuned to match available RAM.
5. **Replaced `write.csv()` with `fwrite()`**, which is typically 10–50× faster for large files.
6. **Added explicit `rm()` + `gc()`** after each year to release memory promptly.

These changes preserve all trained Random Forest models exactly as-is and require no retraining.