 # Diagnosis and Optimization Strategy for Cell-Level GDP Prediction

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (since the loop variable and column share the same name), returning **all rows** instead of the intended subset. This means every year's prediction runs on the entire dataset — massively inflating computation.

2. **Loading Unnecessary Spatial Data:** `prep_data` is loaded via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Heavy Library Loading:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, etc.) are loaded but not used in the prediction step, consuming memory.

4. **Full Data Frame in Memory:** `joined_data` and `pred_db` are likely large data frames held entirely in memory. Subsetting with `dplyr::filter` on a very large data frame each iteration is slower than keyed operations.

5. **Row-by-Row Assignment via Logical Indexing:** `pred_db$consolidated[pred_db$year == year]` scans the entire data frame for every year.

6. **`write.csv` on Large Data:** Writing hundreds of thousands (or millions) of rows with `write.csv` is very slow compared to `fwrite`.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Variable shadowing bug | Rename loop variable to `yr` |
| Unused shapefile load | Remove `st_read()` call |
| Unused libraries | Load only what is needed |
| Slow subsetting | Convert to `data.table`, key by `year` |
| Large prediction batches | Predict in chunks if memory is tight |
| Slow CSV write | Use `data.table::fwrite()` |
| Memory pressure | Explicitly `gc()` after large operations; remove unneeded objects |

---

## Optimized Working R Code

```r
# ── Load only required libraries ──────────────────────────────────
library(data.table)
library(randomForest)

# ── Load pre-trained models ───────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert core data to data.table and key by year ───────────────
# (joined_data and pred_db are expected from the .RData file)
setDT(joined_data)
setDT(pred_db)

setkey(joined_data, year)
setkey(pred_db, year)

# ── Remove objects not needed for prediction ──────────────────────
# Adjust this list to whatever else was loaded from the .RData file
# that is not needed:
# rm(any_other_large_object)
gc()

# ── Initialize output column ─────────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Predict year by year ─────────────────────────────────────────
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Fast keyed subset
    test_set <- joined_data[.(yr)]

    # Predict in chunks to limit peak memory
    n        <- nrow(test_set)
    chunk_sz <- 50000L
    preds    <- numeric(n)

    for (start in seq(1L, n, by = chunk_sz)) {
      end <- min(start + chunk_sz - 1L, n)
      preds[start:end] <- predict(rf_model, newdata = test_set[start:end, ])
    }

    # Assign back using keyed join (fast)
    pred_db[.(yr), consolidated := preds]

    # Free memory from this iteration
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── Write output with fast fwrite ─────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Gains

| Change | Effect |
|---|---|
| Fix `year == year` → `yr` | Correct predictions; avoids processing entire dataset each loop |
| `data.table` with key | ~10–100× faster subsetting and assignment |
| Chunked `predict()` | Caps peak memory; avoids allocating one massive prediction matrix |
| Remove unused `st_read` & libraries | Frees potentially gigabytes of RAM |
| `fwrite` instead of `write.csv` | ~5–20× faster file writing |
| `gc()` per iteration | Returns memory to OS between years |

No models are retrained — only the prediction and I/O pipeline is optimized.