 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary heavy library loading:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`st_read`) that is never used in prediction.** The `prep_data` object may consume gigabytes of RAM for no reason.

3. **Variable masking bug:** `filter(year == year)` inside the loop doesn't work as intended. The column `year` and the loop variable `year` share the same name, so the filter resolves to a tautology (`TRUE` for every row), meaning `test_set` is the *entire* dataset every iteration — massively increasing memory use and prediction time.

4. **`predict()` on the full `randomForest` object with hundreds of thousands of rows** can be memory-intensive. Processing all years sequentially in-memory with a data.frame compounds this.

5. **`joined_data` and `pred_db` may be large data.frames** held entirely in memory simultaneously, along with the list of RF models.

6. **Writing a single massive CSV at the end** can also spike memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries and objects (`prep_data`) | Remove them to free memory |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly, dramatically reducing per-iteration data size |
| Entire `joined_data` and `pred_db` in memory | Convert to `data.table` for fast subset and update-by-reference |
| Large RF `predict` calls | Predict only the correct yearly subset (fixed by the bug fix above) |
| All models in memory at once | Drop each model from the list after use with `rf_models_per_year[[yr]] <- NULL` to free memory progressively |
| Large CSV write | Use `data.table::fwrite` (faster, lower peak memory) |

---

## Optimized R Code

```r
# ── Load only what is needed ──────────────────────────────────────────────
library(randomForest)
library(data.table)
library(tidyverse)        # kept only if joined_data / pred_db are tibbles

# ── Load models and data ─────────────────────────────────────────────────
# Do NOT load the shapefile — it is unused in prediction.
# prep_data <- st_read(...)            # REMOVED

load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert to data.table for speed & memory-efficient update-by-reference
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── Prediction loop ──────────────────────────────────────────────────────
years <- intersect(
  unique(pred_db$year),
  as.numeric(names(rf_models_per_year))
)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correct filter — no more variable-masking bug
  test_idx   <- which(joined_data$year == yr)
  test_set   <- joined_data[test_idx]

  pred_idx   <- which(pred_db$year == yr)

  # Predict only on the true yearly subset
  preds <- predict(rf_model  <- rf_models_per_year[[as.character(yr)]],
                   newdata   = test_set)

  # Update by reference (no copy)
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)

  # Free the model to reclaim memory immediately
  rf_models_per_year[[as.character(yr)]] <- NULL
  rm(test_set, preds, rf_model)
  gc()                 # return memory to the OS

  cat(paste0("  Done. Rows predicted: ", length(pred_idx), "\n"))
}

# Warn about years with no model
no_model_years <- setdiff(unique(pred_db$year), years)
if (length(no_model_years) > 0) {
  cat("Warning: No model found for years:",
      paste(no_model_years, collapse = ", "), "\n")
}

# ── Write results (fwrite is ~5-10× faster than write.csv) ──────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Gains

| Change | Memory Saved | Speed Gained |
|---|---|---|
| Remove unused libraries & `prep_data` | Potentially GBs | Faster startup |
| Fix `year == year` masking bug | **Huge** — predicts on 1/N-th of data instead of all | **N× faster** per iteration |
| `data.table` update-by-reference | Avoids full-copy on each assignment | Faster + lower peak RAM |
| Drop models after use + `gc()` | Frees each RF model (~100s of MB each) | Keeps peak RAM manageable |
| `fwrite` instead of `write.csv` | Lower peak memory | 5–10× faster I/O |

The single most impactful fix is **correcting the variable-masking bug**, which was causing every iteration to predict on the *entire* dataset rather than one year's subset.