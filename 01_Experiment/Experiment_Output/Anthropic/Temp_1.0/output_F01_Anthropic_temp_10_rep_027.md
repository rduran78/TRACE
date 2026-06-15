 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Loading unnecessary data:** `prep_data` is read via `st_read()` but never used, consuming memory for a potentially large spatial dataset.
2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (due to the loop variable sharing the same name), resulting in **no filtering at all** — every row is selected every iteration. This means each call to `predict()` processes the *entire* dataset instead of a single year's subset, massively inflating computation time and memory use.
3. **Whole-dataframe prediction:** Even with the bug fixed, calling `predict()` on hundreds of thousands of rows at once with a Random Forest model can spike memory, especially if the forest is large (many trees, deep trees).
4. **Using a `data.frame` for row-level assignment:** Assigning predictions back into `pred_db$consolidated[pred_db$year == year]` on a very large data.frame is slow due to repeated row-matching.
5. **Writing a massive CSV at the end:** `write.csv()` on a huge data.frame is slow; `data.table::fwrite()` is significantly faster.
6. **All libraries loaded upfront:** Several libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, `zoo`) appear unused in this script, adding to load time and memory footprint.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read()` call | Remove it. |
| Variable shadowing in `filter()` | Rename the loop variable (e.g., `yr`) so `dplyr::filter` correctly subsets. |
| Predicting on full dataset each iteration | Correctly subset per year (bug fix above solves this). |
| Memory spike from large single `predict()` call | Chunk predictions within each year into batches (e.g., 50,000 rows). |
| Slow row assignment on `data.frame` | Convert `pred_db` to a `data.table` and assign by reference. |
| Slow CSV write | Use `data.table::fwrite()`. |
| Unnecessary library loads | Remove unused libraries. |

---

## Optimized R Code

```r
# ── Only load what is actually needed ──────────────────────────────
library(randomForest)
library(tidyverse)
library(data.table)

# ── Load the saved models and associated data ──────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert working tables to data.table for speed ────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Chunked prediction function to limit memory usage ─────────────
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
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

# ── Main prediction loop (note: loop var is 'yr', not 'year') ─────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Correct subset — no variable shadowing
    test_set <- joined_data[year == yr]

    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

    # Assign by reference — fast, no copy
    pred_db[year == yr, consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory each iteration
  gc()
}

# ── Fast write ─────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

- **Removed** unused `st_read()` and unnecessary library calls → lower memory baseline.
- **Fixed the critical bug:** renamed loop variable from `year` to `yr` so `filter`/subsetting works correctly, reducing per-iteration data from the full dataset to only one year's rows.
- **Chunked `predict()`** in batches of 50,000 rows to cap peak memory usage.
- **Switched to `data.table`** for in-place (by-reference) assignment — avoids repeated full-copy overhead.
- **Replaced `write.csv()` with `fwrite()`** for much faster output.
- **Added `gc()`** per iteration to release memory promptly.

No models are retrained; only prediction (inference) is performed.