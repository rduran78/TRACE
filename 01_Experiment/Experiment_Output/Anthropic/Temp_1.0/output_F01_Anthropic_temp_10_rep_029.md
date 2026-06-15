 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several issues in the provided code that cause excessive memory consumption and slow performance:

1. **Loading the entire shapefile unnecessarily.** `prep_data` is loaded via `st_read()` but never used in the prediction loop. Spatial data with hundreds of thousands of geometries is extremely memory-heavy.

2. **Massive library overhead.** Many loaded libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are never used in the prediction step. Each consumes memory.

3. **Variable shadowing bug.** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself—returning **all rows** every iteration instead of the subset for that year. This means every prediction call processes the entire dataset, drastically increasing both time and memory.

4. **Monolithic in-memory prediction.** Calling `predict()` on hundreds of thousands of rows at once for a Random Forest model (which itself can be large) creates a large temporary allocation for every year.

5. **No garbage collection or chunking.** The loop holds all objects in memory simultaneously with no intermediate cleanup.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused shapefile load | Remove `st_read()` entirely |
| Excess libraries | Load only `randomForest`, `data.table`, and `dplyr` (or just `data.table`) |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable, or switch to `data.table` subsetting |
| Large single-pass prediction | Predict in row-chunks (e.g., 50 000 rows at a time) |
| No memory cleanup | Call `gc()` after each year and remove temporary objects |
| CSV write of huge data | Use `data.table::fwrite()` for dramatically faster I/O |

---

## Optimized R Code

```r
# ── Load only what is needed ──
library(randomForest)
library(data.table)

# ── Load pre-trained models and prediction database ──
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for fast subsetting and assignment
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Chunked prediction function ──
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
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
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correct subsetting — no variable-shadowing bug
    test_set <- joined_data[year == yr]

    rf_model <- rf_models_per_year[[yr_char]]
    preds    <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

    pred_db[year == yr, consolidated := preds]

    # Free temporary objects and reclaim memory
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Fast write to disk ──
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

1. **Removed** `st_read()` and all unused library calls → large memory savings.
2. **Fixed the shadowing bug** by renaming the loop variable to `yr` and using `data.table` syntax (`joined_data[year == yr]`), ensuring only the correct year's rows are selected.
3. **Switched to `data.table`** for fast reference-semantics subsetting and in-place column assignment (`:=`), avoiding copies.
4. **Chunked `predict()`** in batches of 50 000 rows to cap peak memory during prediction.
5. **Added `rm()` + `gc()`** after each year to release temporary allocations.
6. **Replaced `write.csv()` with `fwrite()`** for significantly faster output.

No models are retrained; all existing `rf_models_per_year` objects are used as-is.