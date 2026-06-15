 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on the full dataset — massively inflating computation time and producing incorrect results.

3. **Predicting on the entire dataset at once per year:** Even after fixing the bug, calling `predict()` on hundreds of thousands of rows with a large Random Forest model can spike memory usage because the model must route every row through every tree simultaneously.

4. **Using a base `data.frame` for column assignment in a loop:** Repeated assignment to `pred_db$consolidated[pred_db$year == year]` on a very large data.frame is slow due to repeated logical indexing and potential memory copying.

5. **Writing a massive CSV at the end:** `write.csv()` on a very large data.frame is slow; `data.table::fwrite()` is significantly faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused `st_read()` call | Remove it to free memory. |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) to avoid ambiguity. |
| Memory spike during `predict()` | Predict in chunks (batched prediction) to cap peak memory. |
| Slow indexing on `data.frame` | Convert `pred_db` to a `data.table` and use keyed joins or indexed updates. |
| Slow CSV write | Use `data.table::fwrite()`. |
| Unnecessary libraries loaded | Remove unused libraries to reduce overhead. |

The trained Random Forest models are **not retrained** — they are loaded and used as-is.

---

## Optimized R Code

```r
# ── Load only the libraries actually needed ──────────────────────────
library(data.table)
library(randomForest)   # needed for predict() on randomForest objects
library(tidyverse)      # only if joined_data is a tibble; otherwise removable

# ── 1. Load the saved models and associated data ─────────────────────
#    (This loads rf_models_per_year, joined_data, pred_db, etc.)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Convert pred_db and joined_data to data.tables ────────────────
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# ── 3. Define a chunked predict function to limit memory usage ───────
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  
  # Pre-allocate result vector
  preds <- numeric(n)
  
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# ── 4. Loop over years with correct subsetting and chunked prediction ─
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))
  
  yr_char <- as.character(yr)
  
  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]
    
    # Correct filter: use the loop variable, not the column name
    test_set <- joined_data[year == yr]
    
    # Chunked prediction to cap peak memory
    preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)
    
    # Fast indexed update via data.table
    pred_db[year == yr, consolidated := preds]
    
    # Free memory after each year
    rm(preds, test_set)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# ── 5. Write output with fast fwrite ─────────────────────────────────
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

- **Removed** the unused `st_read()` call and unnecessary library loads → frees GBs of RAM.
- **Fixed the variable-shadowing bug** by renaming the loop variable to `yr` and using `data.table` syntax `joined_data[year == yr]`, which correctly references the loop variable.
- **Chunked prediction** (`predict_chunked`) processes rows in batches of 50,000 (tunable), preventing memory from spiking when the model scores hundreds of thousands of rows at once.
- **Converted to `data.table`** for fast subsetting and in-place column updates (no copy-on-modify).
- **Explicit `gc()`** after each year to return memory to the OS promptly.
- **`fwrite()`** replaces `write.csv()` for dramatically faster file output.

These changes make the workflow feasible on a normal machine without retraining any model.