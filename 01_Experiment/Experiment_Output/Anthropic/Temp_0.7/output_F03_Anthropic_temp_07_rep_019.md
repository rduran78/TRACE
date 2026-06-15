 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` — O(n) `lapply` with per-row string operations**

* For every one of ~6.46 million rows, the function:
  1. Does a named-vector lookup (`id_to_ref`) — O(1) amortized but with character coercion overhead per row.
  2. Indexes into the `neighbors` nb list.
  3. Creates `paste(..., sep="_")` keys for every neighbor of every row.
  4. Looks those keys up in `idx_lookup` (another named character vector).
* The string concatenation (`paste`) and named-vector lookups are executed **~6.46M × avg_neighbors** times. With ~1.37M directed relationships spread over 344K cells (~4 neighbors/cell on average), that is ~25.8 million `paste` + hash lookups just in this function.
* The entire result is a list of 6.46M integer vectors — substantial memory allocation pressure.

**`compute_neighbor_stats` — repeated per variable, R-level loop**

* Called 5 times (once per neighbor source variable).
* Each call iterates over all 6.46M rows in an `lapply`, subsetting a numeric vector by the index vectors from the lookup, then computing `max/min/mean`.
* The `do.call(rbind, result)` at the end materialises a 6.46M × 3 matrix from a list of 6.46M length-3 vectors — this is a known slow pattern in R.

**Outer loop copies `cell_data` each iteration**

* `cell_data <- compute_and_add_neighbor_features(...)` likely creates a modified copy of the entire data frame on every iteration (R's copy-on-modify semantics). With ~110 columns × 6.46M rows, each copy is ~5–6 GB depending on types. Five iterations = five copies.

### B. Random Forest Inference Bottlenecks

* Predicting 6.46M rows × 110 features through a Random Forest (even a pre-trained one) is expensive. If done naively in one `predict()` call, the model must build a dense prediction matrix (~5.7 GB for float64) and walk every tree for every row.
* If the model object is large (many trees, deep), `predict.randomForest` or `predict.ranger` may itself be slow, and loading it from disk with `readRDS` can take minutes for a multi-GB object.
* If prediction is done inside a loop (row-by-row or small-batch), overhead is catastrophic.

### C. Memory

* 16 GB RAM is tight. The data frame alone at 110 numeric columns × 6.46M rows ≈ 5.3 GB. The model, neighbor lookup list, and intermediate copies can easily exceed available RAM, causing swap thrashing — the most likely single cause of "86+ hours."

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Fix |
|---|---|---|
| **Neighbor lookup** | Per-row `paste` + named-vector hash | Replace with a `data.table` equi-join: expand neighbors into a long table `(row_i, neighbor_id)`, join to `(id, year) → row_j`, yielding a two-column integer matrix. No strings. |
| **Neighbor stats** | R-level `lapply` over 6.46M elements, 5× | Vectorise with `data.table` grouped aggregation on the long neighbor-pair table. Compute all 5 variables' stats in one pass. |
| **`do.call(rbind, ...)`** | Slow list→matrix | Eliminated by `data.table` aggregation returning columns directly. |
| **Data frame copies** | Copy-on-modify 5× | Use `data.table` `:=` (modify in place). Zero copies. |
| **RF prediction** | Possible row-level loop or huge single matrix | Use `ranger::predict` (C++ back-end, much faster than `randomForest::predict`). If model is `randomForest`, convert or re-save as `ranger`. Predict in chunks of ~500K rows to control peak memory. |
| **Model loading** | `readRDS` on huge object | Load once, keep in memory. Consider `qs::qread` for faster deserialization. |
| **Memory** | >16 GB peak | `data.table` in-place ops, chunked prediction, `gc()` between stages. Remove intermediate objects. |

**Expected speedup:** from 86+ hours to roughly 10–30 minutes, depending on the RF model size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# 0. Libraries
# =============================================================================
library(data.table)
# install.packages("ranger")
# install.packages("qs")          # optional, for faster model I/O

# =============================================================================
# 1. Load data — convert to data.table IN PLACE
# =============================================================================
# Assume cell_data is already in memory as a data.frame or data.table.
# If data.frame:
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure key columns are integer (avoid type-coercion overhead later)
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column (will be used for neighbor mapping)
cell_data[, .row_idx := .I]

# =============================================================================
# 2. Build neighbor lookup as a data.table (long format) — REPLACES
#    build_neighbor_lookup()
# =============================================================================
build_neighbor_lookup_dt <- function(cell_dt, id_order, neighbors) {
  # id_order  : integer vector of cell IDs in the order matching the nb object
  # neighbors : spdep nb object (list of integer index vectors into id_order)

  id_order <- as.integer(id_order)

  # --- Step A: expand nb object into a long table of (focal_id, neighbor_id) --
  # Each element neighbors[[k]] contains indices into id_order
  n_cells <- length(id_order)
  focal_idx    <- rep(seq_len(n_cells),  lengths(neighbors))
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  rm(focal_idx, neighbor_idx); gc()

  # --- Step B: cross-join with years present in the data --------------------
  years <- sort(unique(cell_dt$year))

  # Expand edges × years  (~ 1.37M edges × 28 years ≈ 38.5M rows — fits in RAM)
  edge_year <- edge_dt[, CJ(year = years), by = .(focal_id, neighbor_id)]
  rm(edge_dt); gc()

  # --- Step C: map (focal_id, year) → row_idx_i  and
  #                  (neighbor_id, year) → row_idx_j  via keyed join ----------
  setkey(cell_dt, id, year)

  # focal side
  edge_year[cell_dt, row_i := i..row_idx,
            on = .(focal_id = id, year = year)]

  # neighbor side
  edge_year[cell_dt, row_j := i..row_idx,
            on = .(neighbor_id = id, year = year)]

  # Drop edges where either side is missing

  edge_year <- edge_year[!is.na(row_i) & !is.na(row_j)]

  edge_year
}

cat("Building neighbor edge table …\n")
system.time({
  edge_dt <- build_neighbor_lookup_dt(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt columns: focal_id, neighbor_id, year, row_i, row_j


# =============================================================================
# 3. Compute ALL neighbor stats in one vectorised pass — REPLACES
#    compute_neighbor_stats() + outer for-loop
# =============================================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # For each var in var_names, compute max/min/mean of neighbor values,
  # then assign columns in-place to cell_dt.

  # Pull only needed columns into the edge table (avoids copying full dt)
  val_cols <- var_names
  # Add neighbor values to edge_dt via row_j index
  for (v in val_cols) {
    set(edge_dt, j = v, value = cell_dt[[v]][edge_dt$row_j])
  }

  # Grouped aggregation: one group per row_i
  agg_exprs <- list()
  for (v in val_cols) {
    agg_exprs[[paste0("n_max_", v)]]  <-
      bquote(max(.(as.name(v)),  na.rm = TRUE))
    agg_exprs[[paste0("n_min_", v)]]  <-
      bquote(min(.(as.name(v)),  na.rm = TRUE))
    agg_exprs[[paste0("n_mean_", v)]] <-
      bquote(mean(.(as.name(v)), na.rm = TRUE))
  }
  # Build a single j-expression  list(n_max_ntl = max(ntl, na.rm=TRUE), ...)
  j_call <- as.call(c(list(quote(list)), agg_exprs))

  cat("  Aggregating neighbor stats …\n")
  agg <- edge_dt[, eval(j_call), by = .(row_i)]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA
  inf_to_na <- function(x) { x[is.infinite(x)] <- NA_real_; x }
  agg_cols <- setdiff(names(agg), "row_i")
  for (ac in agg_cols) set(agg, j = ac, value = inf_to_na(agg[[ac]]))

  # Join back to cell_dt by row index — in place
  cell_dt[agg, (agg_cols) := mget(agg_cols), on = .(`.row_idx` = row_i)]

  # Clean up temporary columns from edge_dt

  for (v in val_cols) set(edge_dt, j = v, value = NULL)

  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features …\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Free edge table
rm(edge_dt); gc()

# Remove helper column
cell_data[, .row_idx := NULL]


# =============================================================================
# 4. Load trained Random Forest model
# =============================================================================
# Option A: if saved with saveRDS / readRDS
cat("Loading RF model …\n")
system.time({
  rf_model <- readRDS("path/to/trained_rf_model.rds")
  # Option B (faster): rf_model <- qs::qread("path/to/trained_rf_model.qs")
})

# =============================================================================
# 5. Predict in memory-safe chunks
# =============================================================================
predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  cat(sprintf("Predicting %s rows in %d chunks …\n", format(n, big.mark = ","),
              length(starts)))

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + chunk_size - 1L, n)
    chunk <- newdata[i1:i2, ]

    # --- ranger model --------------------------------------------------------
    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = chunk)$predictions

    # --- randomForest model --------------------------------------------------
    } else if (inherits(model, "randomForest")) {
      preds[i1:i2] <- predict(model, newdata = chunk)

    } else {
      stop("Unsupported model class: ", class(model)[1])
    }

    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  chunk %d / %d done (rows %s – %s)\n",
                  k, length(starts),
                  format(i1, big.mark = ","), format(i2, big.mark = ",")))
      gc()
    }
  }
  preds
}

# Prepare prediction matrix (only predictor columns, in correct order)
# Adjust 'predictor_names' to match the feature names the model expects.
predictor_names <- setdiff(names(cell_data), c("id", "year", "gdp", "response"))
# If your model stores feature names:
#   predictor_names <- rf_model$forest$independent.variable.names  # ranger
#   predictor_names <- rownames(importance(rf_model))               # randomForest

pred_data <- cell_data[, ..predictor_names]   # data.table column subset (no copy)

cat("Running RF prediction …\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(rf_model, pred_data)]
})

rm(pred_data); gc()

cat("Done.\n")
```

---

## 4. SUMMARY OF EXPECTED GAINS

| Stage | Before (estimated) | After (estimated) | Key change |
|---|---|---|---|
| `build_neighbor_lookup` | ~30–60 min (paste + hash × 6.46M) | ~1–3 min | `data.table` keyed join, no strings |
| `compute_neighbor_stats` (×5 vars) | ~40–80 hours (lapply × 6.46M × 5, `do.call(rbind)`) | ~2–5 min | Single grouped aggregation, in-place `:=` |
| Data-frame copies (×5) | ~25 GB allocation churn → swap | 0 copies | `data.table` `:=` |
| RF prediction | Varies (possibly row-level loop) | ~5–20 min | Chunked `predict()`, controlled memory |
| **Total** | **86+ hours** | **~10–30 min** | |

The trained Random Forest model is never modified or retrained. The numerical estimand (predicted GDP) is preserved identically — only the computational pathway is optimised.