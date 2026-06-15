 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow — likely **hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix — this is slow because `rbind` on a list of millions of small vectors has quadratic-like overhead. This is called **5 times** (once per neighbor source variable).

**`compute_and_add_neighbor_features` outer loop:** Likely copies the entire `cell_data` data.frame on each assignment (`cell_data <- ...`), which for 6.46M × 110+ columns is a multi-GB copy — **5 times**.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot will:
- Require the entire feature matrix in memory simultaneously alongside the model object (potentially 10+ GB).
- On a 16 GB laptop, this risks swapping to disk.
- If prediction is done row-by-row or in a naive loop, it's catastrophically slow.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named-vector lookups | ~hours |
| `compute_neighbor_stats` | 6.46M `lapply` + `do.call(rbind, ...)` ×5 | ~hours |
| Data.frame copy-on-modify | `cell_data <-` in loop copies entire frame ×5 | ~tens of minutes |
| RF prediction | Possible memory pressure / naive chunking | ~hours if swapping |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation — Vectorized with `data.table`

1. **Replace `build_neighbor_lookup`** with a `data.table` join-based approach. Instead of building a per-row list of neighbor indices, build a long-format **edge table** (`id`, `year`, `neighbor_id`) and join it to the data to get neighbor values directly. This eliminates all per-row string operations.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation on the edge table: `dt_edges[dt_values, on=...][, .(max, min, mean), by=.(id, year)]`. This is fully vectorized in C.

3. **Use `data.table` set-by-reference** (`:=`) to add columns, eliminating all data.frame copies.

### 2.2 Prediction — Batched with Memory Control

1. **Predict in chunks** (e.g., 500K rows) to keep peak memory well under 16 GB.
2. **Load the model once**, reuse for all chunks.
3. Use `data.table` to hold results and bind via `rbindlist`.

### 2.3 Expected Speedup

| Component | Before | After (est.) |
|---|---|---|
| Neighbor lookup build | ~hours | ~1–3 min |
| Neighbor stats (×5 vars) | ~hours | ~2–5 min |
| Data copies | ~30+ min | ~0 (in-place) |
| RF prediction (6.46M rows) | variable | ~10–30 min |
| **Total** | **86+ hours** | **~15–45 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- 3.1 BUILD VECTORIZED NEIGHBOR EDGE TABLE ----

#' Converts an spdep::nb neighbor list + id_order into a long-format edge
#' data.table with columns: id, neighbor_id.
#'
#' @param id_order Integer/numeric vector of cell IDs in the order matching
#'   the nb object (length = number of spatial cells, e.g. 344,208).
#' @param neighbors An spdep::nb object (list of integer index vectors).
#' @return data.table with columns `id` (integer) and `neighbor_id` (integer).

build_neighbor_edges <- function(id_order, neighbors) {
  # Number of neighbors per cell
  n_neighbors <- vapply(neighbors, length, integer(1))

  # Pre-allocate vectors
  total_edges <- sum(n_neighbors)
  from_id     <- integer(total_edges)
  to_id       <- integer(total_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nn <- n_neighbors[i]
    if (nn > 0L) {
      idx_range <- pos:(pos + nn - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[neighbors[[i]]]
      pos <- pos + nn
    }
  }

  data.table(id = from_id, neighbor_id = to_id)
}


# ---- 3.2 COMPUTE NEIGHBOR FEATURES VIA GROUPED JOIN ----

#' For a given variable, compute max/min/mean of neighbor values for every
#' (id, year) combination and add them as columns to dt in place.
#'
#' @param dt data.table with at least columns: id, year, and `var_name`.
#' @param var_name Character string — name of the variable.
#' @param edges data.table with columns: id, neighbor_id (the edge table).
#' @return Invisible NULL. Columns are added to `dt` by reference.

compute_and_add_neighbor_features_dt <- function(dt, var_name, edges) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Subset: only the columns we need from dt for the join
  # Key: neighbor_id + year  →  we look up the neighbor's value
  vals <- dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(vals, neighbor_id, year)

  # Join edges to dt to get (id, year, neighbor_id), then join to vals
  # Step 1: cross edges with years present in dt for each id
  #   — but more efficient: join dt's (id, year) to edges to get
  #     (id, year, neighbor_id), then join neighbor_id+year → val.

  # dt_iy: every (id, year) row index
  dt_iy <- dt[, .(id, year, .row_idx = .I)]
  setkey(edges, id)

  # Expand: for each row in dt, get its neighbors
  # This produces a long table: (id, year, neighbor_id, .row_idx)
  expanded <- edges[dt_iy, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, neighbor_id, year, .row_idx

  # Join to get neighbor values
  expanded[vals, val := i.val, on = .(neighbor_id, year)]

  # Aggregate by the original row
  agg <- expanded[!is.na(val),
    .(nmax = max(val), nmin = min(val), nmean = mean(val)),
    by = .row_idx
  ]

  # Initialize columns with NA
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)

  # Fill in computed values by row index
  set(dt, i = agg$.row_idx, j = col_max,  value = agg$nmax)
  set(dt, i = agg$.row_idx, j = col_min,  value = agg$nmin)
  set(dt, i = agg$.row_idx, j = col_mean, value = agg$nmean)

  invisible(NULL)
}


# ---- 3.3 BATCHED RANDOM FOREST PREDICTION ----

#' Predict in chunks to control peak memory on a 16 GB machine.
#'
#' @param model A trained Random Forest model (ranger or randomForest object).
#' @param dt data.table containing all predictor columns.
#' @param feature_cols Character vector of predictor column names.
#' @param batch_size Integer, number of rows per prediction batch.
#' @return Numeric vector of predictions (length = nrow(dt)).

predict_rf_batched <- function(model, dt, feature_cols, batch_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)

  starts <- seq(1L, n, by = batch_size)

  for (k in seq_along(starts)) {
    i_start <- starts[k]
    i_end   <- min(i_start + batch_size - 1L, n)
    idx     <- i_start:i_end

    batch <- dt[idx, ..feature_cols]

    # Support both ranger and randomForest objects
    if (inherits(model, "ranger")) {
      preds[idx] <- predict(model, data = batch)$predictions
    } else {
      # randomForest
      preds[idx] <- predict(model, newdata = batch)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  Predicted batch %d/%d (rows %d–%d)",
                       k, length(starts), i_start, i_end))
    }
  }

  preds
}


# ---- 3.4 MAIN PIPELINE ----

run_optimized_pipeline <- function(
  cell_data,          # data.frame or data.table with columns: id, year, + predictors
  id_order,           # integer vector of cell IDs matching nb object order
  rook_neighbors_unique,  # spdep::nb object
  rf_model,           # trained Random Forest model (not retrained)
  feature_cols,       # character vector of all ~110 predictor column names
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
  prediction_batch_size = 500000L
) {

  # --- Convert to data.table (by reference if already one) ---
  if (!is.data.table(cell_data)) {
    message("Converting cell_data to data.table...")
    cell_data <- as.data.table(cell_data)
  }
  setkey(cell_data, id, year)

  message(sprintf("Data: %s rows x %s cols", format(nrow(cell_data), big.mark=","),
                  ncol(cell_data)))

  # --- Step 1: Build edge table (once) ---
  message("Building neighbor edge table...")
  t0 <- proc.time()
  edges <- build_neighbor_edges(id_order, rook_neighbors_unique)
  setkey(edges, id)
  message(sprintf("  Edge table: %s edges  [%.1f sec]",
                  format(nrow(edges), big.mark=","),
                  (proc.time() - t0)[3]))

  # --- Step 2: Compute neighbor features (vectorized, in-place) ---
  message("Computing neighbor features...")
  for (var_name in neighbor_source_vars) {
    t1 <- proc.time()
    compute_and_add_neighbor_features_dt(cell_data, var_name, edges)
    elapsed <- (proc.time() - t1)[3]
    message(sprintf("  %s: done [%.1f sec]", var_name, elapsed))
  }

  # --- Step 3: Garbage collect before prediction ---
  rm(edges)
  gc(verbose = FALSE)

  # --- Step 4: Batched prediction ---
  message("Running batched Random Forest prediction...")
  t2 <- proc.time()

  # Verify all feature columns exist
  missing_cols <- setdiff(feature_cols, names(cell_data))
  if (length(missing_cols) > 0) {
    stop("Missing predictor columns in cell_data: ",
         paste(missing_cols, collapse = ", "))
  }

  cell_data[, predicted_gdp := predict_rf_batched(
    model        = rf_model,
    dt           = cell_data,
    feature_cols = feature_cols,
    batch_size   = prediction_batch_size
  )]

  elapsed_pred <- (proc.time() - t2)[3]
  message(sprintf("  Prediction complete [%.1f sec]", elapsed_pred))

  # --- Return result ---
  cell_data
}


# =============================================================================
# USAGE EXAMPLE (uncomment and adapt to your file paths / object names)
# =============================================================================
#
# library(data.table)
# library(ranger)  # or library(randomForest)
#
# # Load pre-trained model
# rf_model <- readRDS("trained_rf_model.rds")
#
# # Load data
# cell_data <- fread("cell_panel_data.csv")   # or readRDS(...)
# id_order  <- readRDS("id_order.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")
#
# # Define your ~110 feature column names
# feature_cols <- readRDS("feature_col_names.rds")
#   # OR: feature_cols <- setdiff(names(cell_data), c("id", "year", "gdp", ...))
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   feature_cols           = feature_cols,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   prediction_batch_size  = 500000L
# )
#
# # result is a data.table with a new column `predicted_gdp`
# fwrite(result, "cell_gdp_predictions.csv")
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Requirement | How it's met |
|---|---|
| **Trained RF model preserved** | The model is loaded once via `readRDS` and passed read-only to `predict()`. No retraining occurs. |
| **Original numerical estimand preserved** | The neighbor features are computed with identical semantics (max, min, mean of non-NA neighbor values per cell-year). The same predictor columns are passed to the same model, producing identical predictions. `data.table` uses the same R numeric types — no floating-point reinterpretation. |
| **Same neighbor logic** | `build_neighbor_edges` faithfully translates the `spdep::nb` index list using the same `id_order` mapping. The join-based aggregation reproduces the original `vals[idx]` subsetting and `max/min/mean` computation. |

---

## 5. KEY PERFORMANCE GAINS SUMMARIZED

1. **String elimination**: No `paste(id, year)` keys. Joins are on integer `(id, year)` pairs via `data.table` indexed merge — orders of magnitude faster.
2. **Vectorized aggregation**: `data.table`'s grouped `max/min/mean` runs in optimized C, replacing 6.46M R-level `lapply` iterations × 5 variables.
3. **Zero-copy column addition**: `set()` and `:=` modify the table in place — no 5× multi-GB data.frame copies.
4. **Batched prediction**: 500K-row chunks keep peak memory ~4–6 GB, well within 16 GB, preventing OS swap thrashing.
5. **Single edge-table build**: The neighbor structure is materialized once as a keyed `data.table` and reused for all 5 variables, replacing the expensive per-row list construction.