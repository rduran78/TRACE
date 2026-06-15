 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile wastes significant memory.

3. **Loading a monolithic `.RData` file:** The `load()` call likely brings into memory not just the trained models but also the full training data (`joined_data`, `pred_db`, and potentially other large objects), which can exhaust RAM.

4. **Variable masking bug:** `filter(year == year)` inside the loop does not filter correctly — the column name `year` and the loop variable `year` are the same, so the filter evaluates to `TRUE` for every row. This means the entire `joined_data` is passed to `predict()` every iteration, massively inflating computation time and memory use.

5. **Row-by-row assignment in a loop with a data.frame:** Assigning predictions back into `pred_db$consolidated` via logical indexing inside a loop on a very large data.frame is slow.

6. **`randomForest::predict` on hundreds of thousands of rows** can be memory-intensive because it internally allocates matrices proportional to `n_rows × n_trees`.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consuming memory | Remove all unnecessary `library()` calls |
| Unused `prep_data` shapefile in memory | Remove the `st_read()` call entirely |
| Entire `.RData` loads training data into RAM | Save models to a standalone `.rds` file once, then load only that; or selectively remove unneeded objects immediately after `load()` |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Predicting on the full dataset every iteration | Correct the filter so only the relevant year's rows are passed to `predict()` |
| Large single `predict()` call may exceed RAM | Chunk predictions within each year if needed |
| Slow indexed assignment back to data.frame | Use `data.table` for fast keyed assignment |
| Writing a huge CSV is slow | Use `data.table::fwrite()` instead of `write.csv()` |

---

## Optimized R Code

```r
# ------------------------------------------------------------------
# Step 0 (run once): Extract and save only what is needed from the
# large .RData file so future runs are lightweight.
# ------------------------------------------------------------------
# load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# saveRDS(rf_models_per_year, '/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
# saveRDS(pred_db,            '/Volumes/Toshi 1Tb/R_save_files/pred_db.rds')
# saveRDS(joined_data,        '/Volumes/Toshi 1Tb/R_save_files/joined_data.rds')
# rm(list = ls()); gc()
# ------------------------------------------------------------------

library(data.table)
library(randomForest)   # needed only for predict()

# --- Load only the objects required for prediction -----------------
rf_models_per_year <- readRDS('/Volumes/Toshi 1Tb/R_save_files/rf_models_per_year.rds')
pred_db            <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/pred_db.rds'))
joined_data        <- as.data.table(
                        readRDS('/Volumes/Toshi 1Tb/R_save_files/joined_data.rds'))

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# Key both tables by year for fast subsetting
setkey(pred_db,     year)
setkey(joined_data, year)

available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

for (yr in available_years) {                       # renamed loop var
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_num    <- as.numeric(yr)
  test_set  <- joined_data[.(yr_num)]               # fast keyed subset

  rf_model  <- rf_models_per_year[[yr]]

  # --- Chunk predictions to limit peak memory ----------------------
  chunk_size <- 50000L
  n          <- nrow(test_set)
  preds      <- numeric(n)

  for (i in seq(1L, n, by = chunk_size)) {
    idx       <- i:min(i + chunk_size - 1L, n)
    preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
  }

  # Fast update by reference

  pred_db[.(yr_num), consolidated := preds]

  cat(paste0("  Done – ", n, " cells predicted.\n"))
}

# Warn about missing years
missing <- setdiff(as.character(unique(pred_db$year)),
                   names(rf_models_per_year))
if (length(missing) > 0)
  cat("Warning: No models for years:", paste(missing, collapse = ", "), "\n")

# Fast CSV write
fwrite(pred_db, "RF_imputated_db.csv")

cat("Prediction complete. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

1. **Eliminated all unused libraries and the unused shapefile load** — frees potentially gigabytes of RAM.
2. **Separated stored objects into individual `.rds` files** — avoids loading unnecessary training artifacts.
3. **Fixed the variable-masking bug** (`year == year` → keyed subset by `yr_num`) — this alone likely provides the single largest speedup because the original code was accidentally predicting on the *entire* dataset every iteration.
4. **Converted to `data.table`** with keyed subsetting and update-by-reference — eliminates slow logical-index copying.
5. **Chunked `predict()` calls** (50,000 rows at a time) — caps peak memory during prediction regardless of dataset size.
6. **Replaced `write.csv` with `fwrite`** — typically 10–50× faster for large outputs.

These changes preserve the trained Random Forest models exactly as-is, require no retraining, and should make the workflow feasible on a standard personal computer.