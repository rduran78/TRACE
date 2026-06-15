 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction workflow. Spatial data with hundreds of thousands of geometries is extremely memory-intensive.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration — not just the target year. This multiplies computation time by the number of unique years.

4. **Whole-dataframe prediction:** Even without the bug, calling `predict()` on hundreds of thousands of rows at once with a large Random Forest model can spike memory usage because the model must allocate prediction arrays across all trees for all rows simultaneously.

5. **Using a `data.frame` for assignment:** Assigning predictions back into `pred_db$consolidated` row-by-row via logical indexing on a large data.frame is slow. `data.table` would be faster.

6. **Writing a massive CSV:** `write.csv` on a very large data.frame is slow and produces a large file. `data.table::fwrite` is significantly faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read` call entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Memory spike during `predict()` | Predict in chunks (batches) to cap memory usage |
| Slow data.frame indexing | Convert `pred_db` and `joined_data` to `data.table` |
| Slow CSV write | Use `fwrite()` instead of `write.csv()` |
| General | Call `gc()` between years to free memory |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────────
library(randomForest)
library(data.table)

# ── Load the saved models and associated data ────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed ─────────────────────────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Batch-predict helper (caps memory per predict call) ──────────────────
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

# ── Main prediction loop ────────────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))
  
  yr_char <- as.character(yr)
  
  if (yr_char %in% names(rf_models_per_year)) {
    # Correct filter: use the loop variable, not the column vs itself
    test_set <- joined_data[year == yr]
    
    rf_model <- rf_models_per_year[[yr_char]]
    
    preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)
    
    pred_db[year == yr, consolidated := preds]
    
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output efficiently ─────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

- **Removed** all unused library calls and the unused `st_read` shapefile load, reclaiming significant RAM.
- **Fixed the critical bug:** renamed the loop variable to `yr` so that `joined_data[year == yr]` correctly subsets to one year instead of returning the entire dataset every iteration.
- **Chunked prediction:** `predict_in_chunks()` processes rows in batches of 50,000 (tunable), preventing a single massive memory allocation across all trees.
- **Switched to `data.table`:** subsetting with `[year == yr]` and update-by-reference with `:=` are far faster than `dplyr::filter` and base `$<-` on large frames.
- **Used `fwrite()`** instead of `write.csv()` for dramatically faster file output.
- **Explicit `gc()`** after each year frees memory from the prior iteration's temporary objects.