 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning the entire dataset every iteration. This is a critical correctness bug that also amplifies the performance problem.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used, consuming memory for nothing.

3. **Full data in memory as a `data.frame` or `sf` object:** `joined_data` and `pred_db` are likely large `data.frame`/`tibble` objects. Repeated `filter()` and row-indexed assignment (`pred_db$consolidated[pred_db$year == year]`) on hundreds of thousands of rows per year is slow.

4. **`predict.randomForest` on massive data frames:** Passing the entire wide data frame (with columns not needed by the model) forces unnecessary memory copies.

5. **Writing a single monolithic CSV:** `write.csv` on a very large data frame is slow and memory-heavy; `data.table::fwrite` is far faster.

6. **Loading unneeded libraries:** Many loaded packages (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, etc.) are never used, bloating the memory footprint.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so `filter(year == yr)` works correctly |
| Unused `st_read` call | Remove it entirely |
| Unnecessary libraries | Load only what is needed (`data.table`, `randomForest`) |
| Slow subsetting / assignment | Convert to `data.table` and use keyed operations |
| Passing extra columns to `predict()` | Select only the predictor columns the model expects |
| Slow CSV write | Use `data.table::fwrite()` |
| Optional: memory pressure | Process and predict year-by-year, then bind; call `gc()` between iterations |

---

## Optimized Working R Code

```r
# ── Only the libraries actually needed ──────────────────────────
library(data.table)
library(randomForest)

# ── Load pre-trained models ─────────────────────────────────────
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── Convert both data objects to data.table and key by year ─────
setDT(pred_db)
setDT(joined_data)
setkey(pred_db, year)
setkey(joined_data, year)

# ── Initialise the target column ────────────────────────────────
pred_db[, consolidated := NA_real_]

# ── Identify predictor columns from the first available model ───
# (all yearly models share the same feature set)
example_model <- rf_models_per_year[[1L]]
predictor_cols <- rownames(example_model$importance)

# ── Predict year by year ────────────────────────────────────────
years_available <- as.character(names(rf_models_per_year))

for (yr in unique(pred_db$year)) {

  yr_char <- as.character(yr)

  if (yr_char %in% years_available) {

    cat(sprintf("Predicting for year %s …\n", yr_char))

    # Fast keyed subset; select only needed columns
    test_set <- joined_data[.(yr), ..predictor_cols]

    rf_model <- rf_models_per_year[[yr_char]]

    # Predict and assign in place (no copy)
    pred_db[.(yr), consolidated := predict(rf_model, newdata = test_set)]

    # Free temporary objects and reclaim memory
    rm(test_set, rf_model)
    gc()

  } else {
    cat(sprintf("Warning: No model found for year %s\n", yr_char))
  }
}

# ── Write result (fwrite is ~5-10x faster than write.csv) ───────
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

1. **Bug fix:** Loop variable renamed to `yr`; filter now correctly subsets a single year.
2. **Memory:** Removed unused `st_read` call and all unnecessary library loads, cutting idle memory substantially.
3. **Speed – subsetting:** `data.table` keyed join (`[.(yr), ...]`) replaces `dplyr::filter`, which is significantly faster on large tables.
4. **Speed – assignment:** In-place `:=` assignment replaces base-R indexed replacement, avoiding full-column copies.
5. **Speed – predict:** Only the predictor columns are passed to `predict()`, reducing memory allocation inside the Random Forest prediction routine.
6. **Speed – I/O:** `fwrite()` replaces `write.csv()`.
7. **Memory pressure:** `rm()` + `gc()` after each year frees the temporary subset promptly.

No models are retrained; the existing `rf_models_per_year` list is used as-is.