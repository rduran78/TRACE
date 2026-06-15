 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable Masking Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (always `TRUE`), so every iteration loads the *entire* dataset into `test_set` instead of a single year's subset. This is both logically incorrect and extremely wasteful.

2. **Monolithic In-Memory Data:** Loading the full `joined_data` and `pred_db` objects (hundreds of thousands of rows × many columns) into memory at once, then writing a massive CSV at the end, creates peak-memory pressure that can exceed a personal computer's RAM.

3. **`predict.randomForest` on Large Data:** Random Forest prediction scales with `n_rows × n_trees × tree_depth`. Passing the entire dataset (due to the bug) makes this far worse than necessary.

4. **`sf` Geometry Overhead:** `prep_data` is read as an `sf` object (with geometry) but appears unused in the prediction loop — it may be consuming substantial memory for no purpose.

5. **Single-Pass CSV Write:** Writing the entire consolidated result as one CSV at the end means if the process fails partway through, all progress is lost.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable masking bug | Use `.env$year` or rename the loop variable to avoid column-name collision. |
| Full dataset in memory | Convert to `data.table`; subset efficiently by year using keyed joins. |
| Unused `sf` geometry | Remove `prep_data` if unused, or drop geometry before processing. |
| Large prediction batches | Predict in chunks within each year if memory is still tight. |
| No fault tolerance | Write results year-by-year (append mode) so partial progress is saved. |
| CSV write overhead | Use `data.table::fwrite` instead of `write.csv`. |

---

## Optimized R Code

```r
# ── Load only the packages actually needed ──────────────────────────
library(data.table)
library(randomForest)

# ── 1. Load pre-trained models ──────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should provide: rf_models_per_year, joined_data, pred_db
# (Adjust object names if your .RData file differs.)

# ── 2. Remove objects not needed for prediction ─────────────────────
# If prep_data was loaded or is in the .RData, drop it:
if (exists("prep_data")) rm(prep_data)
gc()

# ── 3. Convert to data.table for fast, memory-efficient operations ──
setDT(joined_data)
setDT(pred_db)

# Key both tables on year for fast subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# ── 4. Identify valid years (intersection of data and models) ───────
available_years <- intersect(
  as.character(unique(pred_db$year)),
  names(rf_models_per_year)
)

# ── 5. Identify predictor columns once (avoid sending extra columns) 
# Use the variable names the first model expects:
predictor_vars <- names(rf_models_per_year[[available_years[1]]]$forest$xlevels)
# Fallback: if xlevels is empty (all numeric), use importance row names
if (length(predictor_vars) == 0) {
  predictor_vars <- rownames(importance(rf_models_per_year[[available_years[1]]]))
}

# ── 6. Predict year-by-year, with optional chunking ────────────────
CHUNK_SIZE <- 50000L          # Tune to your machine's RAM

output_file <- "RF_imputed_db.csv"
first_write <- TRUE

for (yr in available_years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_num <- as.numeric(yr)
  rf_model <- rf_models_per_year[[yr]]

  # Fast keyed subset — only the columns the model needs
  test_rows <- joined_data[.(yr_num), ..predictor_vars]

  n <- nrow(test_rows)
  preds <- numeric(n)

  # Chunk the prediction to cap memory use
  starts <- seq(1L, n, by = CHUNK_SIZE)
  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_rows[s:e, ])
  }

  # Write predictions back
  pred_db[.(yr_num), consolidated := preds]

  # ── 7. Append results to disk year-by-year (fault tolerance) ──────
  out_chunk <- pred_db[.(yr_num)]
  fwrite(out_chunk, file = output_file,
         append = !first_write, col.names = first_write)
  first_write <- FALSE

  # Free per-year objects
  rm(test_rows, out_chunk, preds)
  gc()

  cat(sprintf("  → %s rows predicted and written.\n", formatC(n, big.mark = ",")))
}

cat("Done. Output saved to:", output_file, "\n")
```

---

## Summary of Key Changes

| # | Change | Effect |
|---|--------|--------|
| 1 | Fixed `filter(year == year)` bug by using keyed `data.table` subsetting `[.(yr_num)]` | Correct data per year; massive speed-up |
| 2 | Replaced `tidyverse` with `data.table` | Lower memory footprint, faster subset/assign |
| 3 | Selected only predictor columns for `predict()` | Reduces memory copied into the predict call |
| 4 | Added chunked prediction (`CHUNK_SIZE`) | Caps peak memory during `predict()` |
| 5 | Writes results incrementally with `fwrite(..., append=TRUE)` | Fault tolerance + avoids holding full output in memory |
| 6 | Removed unused `prep_data` + explicit `gc()` | Frees RAM immediately |
| 7 | Pre-trained models are **never retrained** | Preserves original models exactly as required |

Adjust `CHUNK_SIZE` (default 50,000 rows) up or down depending on your available RAM. On a machine with 8–16 GB this should run comfortably.