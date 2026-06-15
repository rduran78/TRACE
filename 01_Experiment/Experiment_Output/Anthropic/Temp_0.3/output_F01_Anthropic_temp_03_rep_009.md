 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` contains **all rows** every iteration instead of a single year's subset. This means every prediction pass processes the entire dataset unnecessarily.

3. **Whole-dataframe column assignment in a loop:** `pred_db$consolidated[pred_db$year == year] <- ...` triggers repeated subsetting and copying of a potentially massive data frame on every iteration.

4. **`predict.randomForest` on hundreds of thousands of rows at once** can be memory-intensive because it internally builds a matrix of predictions across all trees.

5. **`joined_data` and `pred_db` may be large `sf` or `tibble` objects** with geometry columns or many unused predictor columns carried in memory.

6. **`write.csv` on a very large data frame** is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `prep_data` load | Remove it |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Large objects in memory | Drop geometry and unused columns before the loop |
| Repeated full-dataframe subsetting | Convert to `data.table` and use keyed joins |
| Large single-pass `predict()` | Chunk predictions within each year if needed |
| Slow `write.csv` | Use `fwrite()` |
| All libraries loaded | Load only what is needed |

---

## Optimized R Code

```r
# ── Load only required libraries ──────────────────────────────────
library(data.table)
library(randomForest)   # needed for predict.randomForest

# ── 1. Load the saved models and prediction data ─────────────────
#    (Do NOT load the heavy shapefile that is never used.)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# ── 2. Identify the predictor names the RF models expect ─────────
#    (Take them from any one of the stored models.)
example_model <- rf_models_per_year[[1]]
rf_features  <- rownames(importance(example_model))

# ── 3. Trim joined_data to only the columns we need ─────────────
#    Drop geometry if it is an sf object; keep only predictors + year + key.
if (inherits(joined_data, "sf")) {
  joined_data <- sf::st_drop_geometry(joined_data)
}
cols_keep_joined <- intersect(names(joined_data), c("year", rf_features))
joined_dt <- as.data.table(joined_data)[, ..cols_keep_joined]
rm(joined_data); gc()

# ── 4. Trim pred_db the same way ────────────────────────────────
if (inherits(pred_db, "sf")) {
  pred_db <- sf::st_drop_geometry(pred_db)
}
pred_dt <- as.data.table(pred_db)
pred_dt[, consolidated := NA_real_]
rm(pred_db); gc()

# ── 5. Key both tables on year for fast subsetting ───────────────
setkey(joined_dt, year)
setkey(pred_dt,   year)

# ── 6. Predict year-by-year, chunked if necessary ───────────────
CHUNK_SIZE <- 50000L          # rows per chunk (tune to your RAM)

years <- intersect(
  unique(pred_dt$year),
  as.numeric(names(rf_models_per_year))
)

for (yr in years) {
  cat(sprintf("Predicting for year %s\n", yr))

  rf_model <- rf_models_per_year[[as.character(yr)]]
  test_dt  <- joined_dt[.(yr)]                       # keyed subset – instant
  n        <- nrow(test_dt)

  if (n == 0L) {
    cat(sprintf("  No test rows for year %s – skipping.\n", yr))
    next
  }

  # Chunked prediction to cap peak memory
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_dt[s:e, ..rf_features])
  }

  pred_dt[.(yr), consolidated := preds]
  rm(test_dt, preds); gc()
}

# ── 7. Write output efficiently ──────────────────────────────────
fwrite(pred_dt, "RF_imputated_db.csv")
cat("Done.\n")
```

---

### Key Changes Summarized

* **Removed** the unused `st_read()` call — saves potentially gigabytes of RAM.
* **Fixed the variable-shadowing bug** by iterating with `yr` and using `data.table` keyed subsetting `[.(yr)]`.
* **Converted to `data.table`** and dropped all columns not needed for prediction — dramatically reduces memory footprint.
* **Chunked `predict()`** calls so that no single call processes more than `CHUNK_SIZE` rows, capping peak memory.
* **Used `fwrite()`** instead of `write.csv()` for much faster output.
* **Freed intermediate objects** with `rm()` and `gc()` after each year.

No models are retrained; the stored `rf_models_per_year` list is used as-is.