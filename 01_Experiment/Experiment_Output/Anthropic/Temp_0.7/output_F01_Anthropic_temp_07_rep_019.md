 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used, consuming memory for nothing.
2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on *all* rows — massively inflating computation time.
3. **Entire `joined_data` held in memory:** With hundreds of thousands of cells × many years × many predictors, this is a large object. Prediction on all of it (due to the bug) makes it worse.
4. **`predict.randomForest` on huge data:** Random Forest prediction scales with `n_rows × n_trees × n_variables`. Doing this repeatedly (and on the wrong-sized subset) is very expensive.
5. **Writing a single monolithic CSV at the end:** For a very large `pred_db`, `write.csv` is slow. `data.table::fwrite` is significantly faster.
6. **No garbage collection or memory management:** All models for all years are loaded simultaneously via `rf_models_per_year`, which may be very large.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read` call | Remove it |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Predicting on full data every iteration | Fix the filter so only one year's rows are predicted |
| Memory pressure from all RF models loaded at once | Optionally load/unload models one at a time if saved separately; at minimum, `gc()` after each year |
| Slow CSV write | Use `data.table::fwrite` |
| `joined_data` as tibble | Convert to `data.table` for faster subsetting |
| No progress feedback on row counts | Add diagnostic `cat()` messages |

---

## Optimized R Code

```r
# ── Only load what is needed ──────────────────────────────────────────────────
library(randomForest)
library(data.table)

# ── Load the saved workspace (models + data) ─────────────────────────────────
# Contains: rf_models_per_year (list of trained RF models),
#           joined_data (predictor data), pred_db (output scaffold)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Remove any large objects from that workspace that are not needed ──────────
# Uncomment and adapt if other large objects were saved in the .RData file:
# rm(object_not_needed); gc()

# ── Convert to data.table for fast keyed subsetting ──────────────────────────
setDT(joined_data)
setDT(pred_db)

# Create a key on year for O(log n) subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# ── Prediction loop (year by year) ───────────────────────────────────────────
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("Warning: No model found for year ", yr, " — skipping.\n"))
    next
  }

  cat(paste0("Predicting for year ", yr, " ... "))

  # ---- Correct, fast subset (no variable-shadowing bug) ----
  test_set <- joined_data[.(yr)]            # keyed lookup, very fast

  rf_model <- rf_models_per_year[[yr_char]]

  preds <- predict(rf_model, newdata = test_set)

  pred_db[.(yr), consolidated := preds]

  cat(paste0(length(preds), " cells done.\n"))

  # ---- Free memory eagerly ----
  rm(test_set, preds)
  gc()
}

# ── Optionally free the models list if no longer needed ──────────────────────
rm(rf_models_per_year, joined_data)
gc()

# ── Fast write ───────────────────────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Impact |
|---|---|---|
| 1 | Removed unused `st_read()` call | Saves memory (large shapefile never loaded) |
| 2 | Renamed loop variable to `yr`; used `joined_data[.(yr)]` | **Fixes the shadowing bug** — the single biggest performance problem |
| 3 | Converted to `data.table` with `setkey` | Subsetting goes from O(n) scan to O(log n) keyed lookup |
| 4 | Added `rm()` + `gc()` inside the loop | Frees prediction vectors each iteration |
| 5 | Replaced `write.csv` with `fwrite` | 5–10× faster file writing |
| 6 | Removed 12 unused `library()` calls | Faster startup, smaller memory footprint |
| 7 | No models retrained | All original RF models preserved and reused as-is |

These changes should make the script feasible on a standard personal computer without any modification to the trained Random Forest models.