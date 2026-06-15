 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (because the loop variable and column share the same name), returning **all rows** instead of the intended subset. This means every iteration predicts on the entire dataset, massively inflating computation time and producing incorrect results.

3. **Whole-dataframe prediction in one call:** Even after fixing the filter bug, calling `predict()` on hundreds of thousands of rows at once can spike memory usage because `randomForest::predict` builds a large matrix internally (n_rows × n_trees).

4. **Using `data.frame` column assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` repeatedly on a large data.frame triggers repeated copies in memory.

5. **Heavy library loading:** Several libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `zoo`) are loaded but not used in the prediction step, consuming memory.

6. **Writing a massive CSV:** `write.csv` on a very large data.frame is slow; `data.table::fwrite` is dramatically faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused spatial data load | Remove `st_read()` call |
| Unused libraries | Remove unnecessary `library()` calls |
| Variable shadowing in `filter()` | Rename loop variable (e.g., `yr`) or use `.env$year` |
| Memory spike from large single `predict()` | Predict in chunks (batched prediction) |
| Slow column assignment in loop | Convert to `data.table` for in-place update by reference |
| Slow CSV write | Use `data.table::fwrite()` |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)
library(dplyr)

# ── Load pre-trained models and data ──────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for fast operations
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Helper: batched predict to limit memory usage ─────────────────────────
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
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

# ── Prediction loop (with corrected filter) ───────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))
  
  yr_char <- as.character(yr)
  
  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]
    
    # Correct filter: use the renamed loop variable 'yr'
    test_set <- joined_data[year == yr]
    
    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)
    
    # In-place update by reference (no copy)
    pred_db[year == yr, consolidated := preds]
    
    # Free memory immediately
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Fast write ────────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

- **Removed** unused `st_read()` and 10 unnecessary library loads → frees significant RAM.
- **Fixed the critical bug:** renamed the loop variable to `yr` so the filter correctly subsets each year.
- **Converted to `data.table`:** enables `:=` update-by-reference (no full-copy overhead) and fast row filtering.
- **Chunked prediction:** `predict_in_chunks()` processes rows in batches of 50,000 (tunable), capping peak memory from `predict()`.
- **Explicit `gc()` per iteration:** releases memory between years.
- **`fwrite()` instead of `write.csv()`:** typically 5–10× faster on large tables.

These changes collectively address both the memory and speed bottlenecks while preserving the pre-trained Random Forest models exactly as-is.