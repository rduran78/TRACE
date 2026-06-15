 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary shapefile read:** `prep_data` is loaded from a large shapefile but never used in the prediction loop. This wastes significant memory.

3. **Variable masking bug:** `filter(year == year)` inside the `for (year in ...)` loop causes the column `year` to be compared to itself (always `TRUE`), meaning **every row** is selected every iteration rather than only the rows for that specific year. This makes every prediction pass operate on the full dataset and produces incorrect results.

4. **Inefficient row-wise assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` inside a loop over years is slow on large data frames due to repeated logical indexing and copy-on-modify behavior.

5. **Loading the entire model file into memory at once:** The `.RData` file likely contains `joined_data`, `rf_models_per_year`, `pred_db`, and possibly other large objects simultaneously, causing a memory spike.

6. **`predict.randomForest` on hundreds of thousands of rows** can be memory-intensive; chunked prediction helps.

7. **Writing a massive CSV with `write.csv`** is slower than `data.table::fwrite`.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries & objects | Remove them; only load what is needed |
| `prep_data` shapefile | Do not load it |
| Variable masking bug | Rename loop variable (e.g., `yr`) so the filter works correctly |
| Full data copied each iteration | Use `data.table` for in-place assignment by reference |
| Memory spike from `.RData` | After loading, remove unneeded objects and call `gc()` |
| Large single-pass `predict()` | Chunk predictions within each year to cap memory |
| Slow CSV write | Use `data.table::fwrite` |

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── Load the saved workspace (models + data) ────────────────────────
load("/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData")

# Convert the two key objects to data.tables for speed
setDT(pred_db)
setDT(joined_data)

# Free every other object that came from the .RData file
# (adjust the keep list if your file contains different names)
keep_objs <- c("pred_db", "joined_data", "rf_models_per_year")
rm(list = setdiff(ls(), keep_objs))
gc()

# ── Pre-allocate the output column ──────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Chunked prediction parameters ──────────────────────────────────
CHUNK_SIZE <- 50000L   # rows per chunk; tune to your RAM

# ── Prediction loop (year by year, chunked) ─────────────────────────
years <- sort(unique(pred_db$year))

for (yr in years) {
  cat(sprintf("Predicting for year %s …\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s – skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct filter: use the renamed loop variable to avoid masking
  test_idx <- which(joined_data$year == yr)
  test_set <- joined_data[test_idx]

  pred_idx <- which(pred_db$year == yr)

  # Sanity check
  if (nrow(test_set) != length(pred_idx)) {
    cat(sprintf("  Warning: row count mismatch (joined_data: %d, pred_db: %d)\n",
                nrow(test_set), length(pred_idx)))
  }

  # Chunked predict to limit peak memory
  n        <- nrow(test_set)
  n_chunks <- ceiling(n / CHUNK_SIZE)
  preds    <- numeric(n)

  for (ch in seq_len(n_chunks)) {
    row_start <- (ch - 1L) * CHUNK_SIZE + 1L
    row_end   <- min(ch * CHUNK_SIZE, n)
    preds[row_start:row_end] <- predict(rf_model,
                                        newdata = test_set[row_start:row_end])
  }

  # In-place assignment by reference (no copy)
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)

  # Free per-year temporaries
  rm(test_set, preds, rf_model)
  gc()
}

# ── Write output efficiently ────────────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

| # | What changed | Why |
|---|---|---|
| 1 | Removed 12 unused `library()` calls and the shapefile read | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` | Fixes the `filter(year == year)` self-comparison bug |
| 3 | Switched from `dplyr` / base data.frame to `data.table` | `set()` updates by reference with zero copies |
| 4 | Added chunked `predict()` with configurable `CHUNK_SIZE` | Caps peak memory during prediction |
| 5 | Explicit `rm()` + `gc()` after loading and inside the loop | Reclaims memory between years |
| 6 | Replaced `write.csv` with `fwrite` | Much faster for large tables |
| 7 | **No model retraining** | All `rf_models_per_year` are loaded and used as-is |

These changes should make the script feasible on a machine with 8–16 GB of RAM without altering any trained model.