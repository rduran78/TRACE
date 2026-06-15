 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (not to the loop variable), so `test_set` will always equal the entire `joined_data` dataframe. This means every iteration predicts on *all* rows instead of one year's subset, massively inflating computation.

2. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Heavy library loading:** Several libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`) are loaded but not used in the prediction step, consuming memory.

4. **Data frame row assignment in a loop:** Assigning predictions into a large data frame row-by-row-subset (`pred_db$consolidated[pred_db$year == year]`) inside a loop is slow due to repeated logical indexing on a large object.

5. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive, but is unavoidable; however, doing it on the *full* dataset every iteration (due to the bug) compounds the problem.

6. **`write.csv`** on a very large data frame is slow; `data.table::fwrite` is much faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Unused shapefile load | Remove `st_read()` call |
| Unnecessary libraries | Load only what is needed |
| Slow row-subset assignment | Collect predictions in a list, then bind once |
| Slow CSV write | Use `data.table::fwrite()` |
| Optional: memory pressure | Use `gc()` between years; convert to `data.table` |

---

## Optimized R Code

```r
# ── Load only required packages ──────────────────────────────────────────────
library(randomForest)
library(data.table)
library(dplyr)            # for filter / bind_rows

# ── Load pre-trained models and prediction database ──────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert to data.table for speed (if not already)
setDT(pred_db)
setDT(joined_data)

# ── Prediction loop ─────────────────────────────────────────────────────────
years_to_predict <- unique(pred_db$year)
results_list     <- vector("list", length(years_to_predict))

for (i in seq_along(years_to_predict)) {
  yr <- years_to_predict[i]
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # ---- FIX: subset correctly using the loop variable, not the column ----
    test_set <- joined_data[year == yr]

    rf_model    <- rf_models_per_year[[yr_char]]
    predictions <- predict(rf_model, newdata = test_set)

    results_list[[i]] <- data.table(
      row_idx     = which(pred_db$year == yr),
      consolidated = predictions
    )

    # Free memory after each year
    rm(test_set, predictions, rf_model)
    gc()
  } else {
    cat(sprintf("  Warning: No model found for year %s\n", yr))
  }
}

# ── Combine results and assign in one vectorised operation ───────────────────
results <- rbindlist(results_list)
pred_db[, consolidated := NA_real_]
pred_db$consolidated[results$row_idx] <- results$consolidated

# ── Write output with fast writer ────────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed `st_read()` for unused shapefile | Saves potentially gigabytes of RAM |
| 2 | Removed 10 unused `library()` calls | Reduces memory footprint and load time |
| 3 | Fixed `filter(year == year)` → `joined_data[year == yr]` | **Critical bug fix** — predicts only the correct year's rows |
| 4 | Collected predictions in a list, then `rbindlist` + single vectorised assignment | Avoids repeated large logical scans of `pred_db` |
| 5 | Added `rm()` + `gc()` after each year | Frees memory between iterations |
| 6 | Replaced `write.csv` with `fwrite` | ~5–10× faster file writing |
| 7 | No models are retrained | Requirement preserved |

These changes should make the prediction loop feasible on a standard personal computer with moderate RAM (8–16 GB).