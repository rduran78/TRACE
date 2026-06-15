 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning *all* rows every iteration. This means every prediction pass processes the entire dataset instead of a single year's subset, massively inflating computation time and memory use.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full data kept in memory as a data.frame:** `pred_db` and `joined_data` are likely large `data.frame`/`tibble` objects. Subsetting and assigning into them row-by-row-by-year with tidyverse filtering is slow for hundreds of thousands of rows.

4. **Loading the entire `.RData` workspace:** This pulls every object (training data, intermediate objects, etc.) into RAM even though only `rf_models_per_year` and the prediction inputs are needed.

5. **Single-pass CSV write of a potentially huge data.frame:** `write.csv` on a very large frame is slow; `data.table::fwrite` is much faster.

6. **No garbage collection or memory management** between yearly iterations.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) so dplyr distinguishes the column from the external variable. |
| Unused shapefile load | Remove the `st_read()` call. |
| Entire `.RData` in memory | Load only the needed objects, or immediately remove unneeded ones after loading. |
| Slow subsetting / assignment | Convert `pred_db` and `joined_data` to `data.table`; use keyed subsetting and update-by-reference. |
| Slow CSV write | Use `data.table::fwrite`. |
| Memory pressure | Call `gc()` after each year; optionally process in chunks. |

---

## Optimized R Code

```r
library(randomForest)
library(data.table)

# ── 1. Load only what is needed ──────────────────────────────────────────────
# Load the saved workspace (contains rf_models_per_year, pred_db, joined_data)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove any large objects that are NOT needed for prediction
# Adjust this list to match your actual workspace contents
objects_to_keep <- c("rf_models_per_year", "pred_db", "joined_data")
rm(list = setdiff(ls(), objects_to_keep))
gc()

# ── 2. Convert to data.table for fast keyed operations ───────────────────────
setDT(pred_db)
setDT(joined_data)

# Create keyed index on year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Initialise the target column
pred_db[, consolidated := NA_real_]

# ── 3. Predict year-by-year ──────────────────────────────────────────────────
years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Fast keyed subset — only rows for this year
  test_set <- joined_data[.(as.numeric(yr))]

  # Predict using the pre-trained model (no retraining)
  preds <- predict(rf_models_per_year[[yr]], newdata = test_set)

  # Update by reference — no copy of the whole table
  pred_db[.(as.numeric(yr)), consolidated := preds]

  # Free memory immediately
  rm(test_set, preds)
  gc()
}

# Warn about years with no model
missing <- setdiff(as.character(unique(pred_db$year)), names(rf_models_per_year))
if (length(missing) > 0) {
  cat("Warning: No model found for year(s):", paste(missing, collapse = ", "), "\n")
}

# ── 4. Write output efficiently ──────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

### Summary of Key Changes

- **Bug fix:** The loop variable is now `yr`, eliminating the `year == year` self-comparison that caused every year to process the full dataset.
- **`data.table` keyed operations:** Subsetting with `[.(value)]` and update-by-reference (`:=`) avoid copying large objects.
- **Removed unused `st_read()`** to free significant memory.
- **Explicit `rm()` + `gc()`** after each iteration to keep peak memory low.
- **`fwrite`** replaces `write.csv` for substantially faster output.
- **No models are retrained;** only `predict()` is called on existing `rf_models_per_year`.