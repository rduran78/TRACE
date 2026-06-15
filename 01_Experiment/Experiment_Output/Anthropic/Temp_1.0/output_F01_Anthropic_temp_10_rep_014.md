 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (since the loop variable and the column share the same name), returning *all* rows every iteration instead of the intended subset. This is both a correctness bug and a massive performance drain.

2. **Loading unnecessary geospatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Tidyverse overhead on large data:** `dplyr::filter()` on a data frame with hundreds of thousands of rows per year, repeated for every year, is slower than index-based or `data.table` subsetting.

4. **Whole-dataframe assignment in a loop:** `pred_db$consolidated[pred_db$year == year]` forces a full logical scan of the entire data frame on every iteration.

5. **`write.csv()` on a very large data frame:** This is slow; `data.table::fwrite()` is dramatically faster.

6. **All predictor columns loaded into memory at once:** If `joined_data` carries geometry or unused columns, they consume memory for no benefit during `predict()`.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing | Rename the loop variable (e.g., `yr`) |
| Unused shapefile | Remove the `st_read()` call |
| Slow subsetting | Convert to `data.table` and use keyed subsetting |
| Geometry columns in prediction data | Drop geometry (`st_drop_geometry`) if present |
| Slow CSV write | Use `fwrite()` |
| Memory pressure | Use `gc()` between years; predict in place via `data.table` set-by-reference |
| Unnecessary libraries | Remove unused libraries to reduce namespace overhead |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(randomForest)
library(data.table)
library(sf)            # only if joined_data is an sf object

# ── 1. Load the pre-trained models ──────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Drop geometry if joined_data / pred_db are sf objects ────────
if (inherits(joined_data, "sf")) {
  joined_data <- st_drop_geometry(joined_data)
}
if (inherits(pred_db, "sf")) {
  pred_db <- st_drop_geometry(pred_db)
}

# ── 3. Convert to data.table for fast keyed operations ──────────────
setDT(joined_data)
setDT(pred_db)

# Create the target column
pred_db[, consolidated := NA_real_]

# Key both tables on year for O(log n) subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# ── 4. Get the list of model years once ─────────────────────────────
model_years <- names(rf_models_per_year)

# ── 5. Prediction loop (renamed iterator to avoid shadowing) ────────
unique_years <- unique(pred_db$year)

for (yr in unique_years) {

  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% model_years) {

    # Fast keyed subset — returns only rows for this year
    test_set <- joined_data[.(yr)]

    # Predict
    preds <- predict(rf_models_per_year[[yr_char]], newdata = test_set)

    # Assign by reference (no full-column copy)
    pred_db[.(yr), consolidated := preds]

  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Free memory after each year
  gc()
}

# ── 6. Fast write ───────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Gains

| Area | Before | After |
|---|---|---|
| **Correctness** | `filter(year == year)` returns all rows (bug) | `joined_data[.(yr)]` subsets correctly |
| **Subsetting speed** | Full logical scan via `dplyr` each iteration | Keyed `data.table` binary-search lookup |
| **Memory (geometry)** | Geometry kept in RAM during prediction | Dropped before prediction |
| **Memory (per-iteration)** | No cleanup | `gc()` after each year |
| **Assignment** | Copy-on-modify full column each loop | `data.table` set-by-reference (`:=`) |
| **CSV write** | `write.csv()` (slow, single-threaded) | `fwrite()` (fast, multi-threaded) |
| **Unused data** | Large shapefile loaded and never used | Removed |
| **Models** | Preserved (not retrained) | Preserved (not retrained) ✔ |