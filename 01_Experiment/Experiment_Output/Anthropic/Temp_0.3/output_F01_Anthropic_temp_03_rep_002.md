 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — massively inflating computation time and producing incorrect predictions.

4. **Monolithic `predict()` on huge data frames:** Calling `predict()` on hundreds of thousands of rows at once with a large Random Forest model can spike memory usage because the forest must route every observation through every tree simultaneously.

5. **Using a `data.frame` for row-level assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` with a standard `data.frame` is slow; `data.table` set-by-reference is far faster.

6. **Writing a potentially enormous CSV at the end:** A single `write.csv()` call on a very large frame can be slow and memory-hungry.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries & objects | Remove them to free memory |
| `prep_data` never used | Don't load it |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Large single `predict()` call | Predict in chunks (batches) to cap peak memory |
| Slow row assignment | Convert `pred_db` to `data.table` and use set-by-reference |
| Large CSV write | Use `data.table::fwrite()` for fast, memory-efficient output |
| General | Call `gc()` between years to reclaim memory |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load pre-trained models and prediction database ───────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: pred_db, joined_data, rf_models_per_year

# ── Convert to data.table for fast operations ─────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Batch-predict helper (caps peak memory) ───────────────────────
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  preds
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
  
  rf_model <- rf_models_per_year[[yr_char]]
  
  # Correct filter: loop variable 'yr', not column name 'year'
  test_idx <- which(joined_data$year == yr)
  test_set <- joined_data[test_idx]
  
  pred_idx <- which(pred_db$year == yr)
  
  if (nrow(test_set) != length(pred_idx)) {
    cat(sprintf("  Warning: row count mismatch (joined_data: %d, pred_db: %d)\n",
                nrow(test_set), length(pred_idx)))
  }
  
  # Chunk-wise prediction to limit memory spikes
  preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)
  
  # Fast set-by-reference assignment
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)
  
  # Free memory before next iteration
  rm(test_set, preds)
  gc()
}
  
# ── Write output efficiently ──────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

- **Removed** 12 unused library calls and the unused `st_read()`, reclaiming significant RAM.
- **Fixed the critical bug:** renamed the loop iterator to `yr` so `joined_data$year == yr` filters correctly instead of returning all rows every iteration.
- **Chunked prediction:** `predict_in_chunks()` processes 50,000 rows at a time (tunable), preventing memory spikes from routing hundreds of thousands of observations through the forest simultaneously.
- **`data.table` throughout:** `set()` for O(1) by-reference column updates; `fwrite()` for fast, low-memory CSV output.
- **Explicit `gc()`** after each year to return memory to the OS promptly.
- **No models are retrained** — only `predict()` is called on the existing `rf_models_per_year`.