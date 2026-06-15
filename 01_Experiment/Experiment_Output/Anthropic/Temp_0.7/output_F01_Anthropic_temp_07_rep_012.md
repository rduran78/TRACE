 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` contains **all rows** every iteration rather than the subset for that year. This means every prediction pass processes the entire dataset instead of one year's slice, massively inflating computation.

3. **Full dataset loaded into memory at once:** The `joined_data` and `pred_db` objects (hundreds of thousands of rows × many predictors) are held entirely in memory simultaneously, which may exceed RAM on a personal machine.

4. **`randomForest::predict` on huge data frames is slow and memory-hungry:** Predicting hundreds of thousands of rows through a large Random Forest ensemble in a single call allocates large intermediate matrices.

5. **Writing a massive CSV at the end** with `write.csv` is slow for large data; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused shapefile loaded | Remove the `st_read()` call |
| Variable shadowing bug (`year == year`) | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Entire dataset in memory | Use `data.table` for `pred_db` and `joined_data`; process year-by-year slices |
| Large single `predict()` call | Chunk predictions within each year into batches (e.g., 50 000 rows) to cap peak memory |
| Slow CSV write | Use `data.table::fwrite` |
| Many unused libraries loaded | Remove unused libraries to reduce overhead |
| No garbage collection between years | Call `gc()` after each year to free memory |

The trained Random Forest models (`rf_models_per_year`) are **preserved and never retrained**.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(data.table)
library(randomForest)   # needed for predict()
library(dplyr)          # filter / select (lightweight use)

# ── Configuration ────────────────────────────────────────────────────
CHUNK_SIZE  <- 50000L   # rows per prediction batch – tune to your RAM
MODEL_PATH  <- "/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData"
OUTPUT_PATH <- "RF_imputated_db.csv"

# ── Load saved workspace (contains rf_models_per_year, pred_db,
#    joined_data, and possibly other objects) ─────────────────────────
load(MODEL_PATH)

# ── Convert to data.table for speed / lower memory ──────────────────
if (!is.data.table(pred_db))      setDT(pred_db)
if (!is.data.table(joined_data))  setDT(joined_data)

# ── Initialise the output column ─────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Predict year by year ─────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, " ... "))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("WARNING – no model found, skipping.\n")
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # ---- Correct filter: use the loop variable, not the column --------
  test_set <- joined_data[year == yr]

  n <- nrow(test_set)
  preds <- numeric(n)

  # ---- Chunked prediction to limit peak memory ----------------------
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # ---- Write predictions back into pred_db --------------------------
  pred_db[year == yr, consolidated := preds]

  # ---- Free temporary objects and reclaim memory --------------------
  rm(test_set, preds)
  gc()

  cat(paste0(n, " cells done.\n"))
}

# ── Fast write ───────────────────────────────────────────────────────
fwrite(pred_db, OUTPUT_PATH)
cat("Output written to", OUTPUT_PATH, "\n")
```

---

## Summary of Key Changes

1. **Removed** the unused `st_read()` call and all unnecessary library loads.
2. **Fixed the filtering bug** by renaming the loop variable to `yr` so `joined_data[year == yr]` correctly subsets one year.
3. **Converted to `data.table`** for faster subsetting and lower memory footprint.
4. **Chunked `predict()` calls** (default 50 000 rows per batch) so intermediate matrices stay small.
5. **Added `gc()`** after each year to release memory promptly.
6. **Replaced `write.csv` with `fwrite`** for significantly faster output.
7. **All trained models are untouched**—no retraining occurs.