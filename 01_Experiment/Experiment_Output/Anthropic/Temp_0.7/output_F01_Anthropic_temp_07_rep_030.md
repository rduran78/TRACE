 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Scoping Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration, not just the target year. This massively inflates the prediction workload.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full Data in Memory:** `joined_data` and `pred_db` are both held entirely in memory. If they are large `sf` or `data.frame` objects with geometry columns, memory use is amplified.

4. **Row-Level Assignment in a Loop:** `pred_db$consolidated[pred_db$year == year]` performs a full-column logical scan on every iteration.

5. **`randomForest::predict` on Massive Data:** Predicting hundreds of thousands of rows through a large Random Forest in one call can spike memory because the method internally allocates prediction matrices across all trees.

6. **CSV Output of Huge Data:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Filter bug (`year == year`) | Use a distinct loop variable name (e.g., `yr`) |
| Unused shapefile load | Remove `st_read` call |
| High memory from geometry columns | Drop geometry before prediction |
| Large single-call predict | Predict in chunks (batches) |
| Slow row assignment | Use `data.table` keyed joins |
| Slow CSV write | Use `data.table::fwrite` |
| All years in memory at once | Process year-by-year, writing results incrementally (optional) |

---

## Optimized Working R Code

```r
library(randomForest)
library(data.table)
library(sf)

# ------------------------------------------------------------------
# 1. Load only what is needed (do NOT load the unused shapefile)
# ------------------------------------------------------------------
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: pred_db, joined_data, rf_models_per_year

# ------------------------------------------------------------------
# 2. Convert to data.table for speed; drop geometry if present
# ------------------------------------------------------------------
if (inherits(joined_data, "sf")) {
  joined_data <- st_drop_geometry(joined_data)
}
if (inherits(pred_db, "sf")) {
  pred_db <- st_drop_geometry(pred_db)
}

setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# ------------------------------------------------------------------
# 3. Key joined_data by year for fast subsetting
# ------------------------------------------------------------------
setkey(joined_data, year)
setkey(pred_db, year)

# ------------------------------------------------------------------
# 4. Chunked prediction function to limit peak memory
# ------------------------------------------------------------------
predict_in_chunks <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (i in seq_along(starts)) {
    idx <- starts[i]:min(starts[i] + chunk_size - 1L, n)
    preds[idx] <- predict(model, newdata = newdata[idx, ])
  }
  return(preds)
}

# ------------------------------------------------------------------
# 5. Predict year-by-year with correct variable scoping
# ------------------------------------------------------------------
years_to_predict <- unique(pred_db$year)

for (yr in years_to_predict) {                        # 'yr' avoids the scoping bug
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  # Fast keyed subset (only rows for this year)
  test_set <- joined_data[.(yr)]

  rf_model <- rf_models_per_year[[yr_char]]

  preds <- predict_in_chunks(rf_model, test_set, chunk_size = 50000L)

  pred_db[.(yr), consolidated := preds]

  # Optional: free memory each iteration
  rm(test_set, preds)
  gc()
}

# ------------------------------------------------------------------
# 6. Fast file write
# ------------------------------------------------------------------
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

- **Bug fix:** Renamed loop variable to `yr` so the filter correctly subsets each year.
- **Removed** the unused `st_read` call, freeing substantial memory.
- **Dropped geometry** columns before prediction (not needed by Random Forest).
- **Converted to `data.table`** with keyed indexing for fast year-based subsetting and assignment.
- **Chunked prediction** caps peak memory by feeding the model ≤50,000 rows at a time.
- **`gc()` per iteration** releases memory between years.
- **`fwrite`** replaces `write.csv` for dramatically faster output.

No models are retrained; the existing `rf_models_per_year` list is used as-is.