 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning *all* rows every iteration. This means every year's prediction runs on the entire dataset — massively inflating both memory use and compute time.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes significant memory for no benefit.

3. **Full data frame kept in memory:** `joined_data` and `pred_db` may both be very large data frames with geometry columns and many predictor variables. Holding everything in memory at once is wasteful when predictions are done year-by-year.

4. **Data frame column assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a large data frame inside a loop triggers repeated full-column scans and copy-on-modify behavior in base R.

5. **Writing a single massive CSV at the end:** For hundreds of thousands of rows × many columns, `write.csv()` is slow and memory-hungry.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) so dplyr filters correctly. |
| Unused `st_read()` call | Remove it entirely. |
| Entire dataset in memory | Convert to `data.table`; subset by year efficiently using keyed joins. Drop unneeded columns (especially geometry) before prediction. |
| Slow column assignment in loop | Use `data.table`'s set-by-reference (`:=`) with keyed subsetting. |
| Large CSV write | Use `data.table::fwrite()` for dramatically faster I/O. |
| Optional: memory pressure | Process and write results year-by-year (append mode) so peak memory is reduced. |

---

## Optimized Working R Code

```r
library(randomForest)
library(data.table)

# ------------------------------------------------------------------
# 1. Load only what is needed
# ------------------------------------------------------------------
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: pred_db, joined_data, rf_models_per_year

# 2. Convert to data.table for fast, memory-efficient operations
#    Drop any sf geometry if present — not needed for prediction
if (inherits(joined_data, "sf")) {
  joined_data <- as.data.table(sf::st_drop_geometry(joined_data))
} else {
  setDT(joined_data)
}

if (inherits(pred_db, "sf")) {
  pred_db <- as.data.table(sf::st_drop_geometry(pred_db))
} else {
  setDT(pred_db)
}

# 3. Key both tables on year for fast subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# 4. Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# 5. Identify which years actually have a trained model
available_years <- as.integer(names(rf_models_per_year))
loop_years      <- intersect(unique(pred_db$year), available_years)

# 6. Predict year-by-year using data.table keyed subsetting
for (yr in loop_years) {
  cat(sprintf("Predicting for year %d ...\n", yr))

  # Fast keyed subset — only the rows for this year
  test_set <- joined_data[.(yr)]

  rf_model <- rf_models_per_year[[as.character(yr)]]

  # Predict and assign by reference (no copy of the whole column)
  preds <- predict(rf_model, newdata = test_set)
  pred_db[.(yr), consolidated := preds]

  # Free memory immediately
  rm(test_set, preds)
  gc()
}

# 7. Warn about years with no model
missing <- setdiff(unique(pred_db$year), available_years)
if (length(missing) > 0) {
  cat("Warning: No model found for years:", paste(missing, collapse = ", "), "\n")
}

# 8. Fast write
data.table::fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

- **Bug fix:** The year-filtering shadowing issue is resolved by using a distinctly named loop variable (`yr`) and `data.table` keyed subsetting (`[.(yr)]`).
- **Memory savings:** Geometry columns are dropped; `data.table` operates by reference instead of copying; `gc()` is called each iteration.
- **Speed gains:** `data.table` keyed lookups are O(log n) instead of full scans; `fwrite()` is multithreaded and far faster than `write.csv()`.
- **Models untouched:** No Random Forest model is retrained or modified. Only `predict()` is called.