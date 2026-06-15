 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable Masking Bug:** `filter(year == year)` compares the column `year` to itself (always `TRUE`), so every iteration loads the *entire* dataset into `test_set` instead of just one year's slice. This means every `predict()` call runs on the full dataset — massively inflating memory use and compute time.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes memory.

3. **Full Data Frame in Memory:** `joined_data` and `pred_db` may both be very large `data.frame`/`tibble` objects. Operating on them with `dplyr::filter` and row-indexed assignment (`pred_db$consolidated[...]`) is slow and memory-hungry for hundreds of thousands of rows.

4. **Monolithic CSV Write:** Writing a very large CSV at the end with `write.csv` is slower than alternatives.

5. **All RF Models Loaded at Once:** `rf_models_per_year` is a list of all yearly Random Forest models. If memory is tight, holding all of them simultaneously is costly.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable masking bug | Use `.env$year` or rename the loop variable to avoid column/variable collision. |
| Unused shapefile | Remove the `st_read()` call. |
| Slow subsetting & assignment | Convert `joined_data` and `pred_db` to `data.table`; key by `year` for fast subset and update-by-reference. |
| All models in memory | Load the `.RData` file, then process one year at a time, removing each model from the list after use via `rf_models_per_year[[yr]] <- NULL` to free memory. |
| Large CSV write | Use `data.table::fwrite()` instead of `write.csv()`. |
| Optional: parallel predict | For very large single-year slices, chunk the `predict()` call to cap peak memory. |

None of these changes retrain or alter the Random Forest models.

---

## Optimized R Code

```r
# ── Load only the packages actually needed ──────────────────────────
library(data.table)
library(randomForest)

# ── 1. Load pre-trained models and prediction database ──────────────
#    (Do NOT load the unused shapefile)
load("/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData")

# ── 2. Convert to data.table for speed and memory efficiency ────────
setDT(pred_db)
setDT(joined_data)

# Key both tables by year for fast subsetting
setkey(pred_db,    year)
setkey(joined_data, year)

# Pre-allocate the output column (numeric, NA)
pred_db[, consolidated := NA_real_]

# ── 3. Predict year-by-year ─────────────────────────────────────────
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("Warning: No model found for year", yr, "\n")
    next
  }

  cat("Predicting for year", yr, "...")

  # Fast keyed subset – only this year's rows
  test_set <- joined_data[.(yr)]

  # Retrieve the model
  rf_model <- rf_models_per_year[[yr_char]]

  # ---- Optional: chunk predict() to limit peak memory ----
  n        <- nrow(test_set)
  chunk_sz <- 50000L
  preds    <- numeric(n)

  if (n <= chunk_sz) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_sz)
    for (s in starts) {
      e <- min(s + chunk_sz - 1L, n)
      preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
    }
  }
  # ---- End optional chunking --------------------------------

  # Update by reference – no copy of the whole table
  pred_db[.(yr), consolidated := preds]

  # Free the model to reclaim memory immediately
  rf_models_per_year[[yr_char]] <- NULL
  rm(rf_model, test_set, preds)
  gc()                 # return memory to the OS


  cat(" done.\n")
}

# ── 4. Write results efficiently ────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("All predictions written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Bug fix:** The loop variable was renamed to `yr` so `joined_data[.(yr)]` correctly subsets to a single year instead of matching the column to itself.
2. **Memory:** Each model is removed from the list (`<- NULL`) right after use; `gc()` is called each iteration.
3. **Speed:** `data.table` keyed joins replace `dplyr::filter` and vectorized row indexing; `fwrite` replaces `write.csv`.
4. **Optional chunking:** `predict()` is split into 50 000-row chunks to cap peak memory during prediction on very large year-slices.
5. **Models are never retrained** — only loaded and used for inference, then freed.