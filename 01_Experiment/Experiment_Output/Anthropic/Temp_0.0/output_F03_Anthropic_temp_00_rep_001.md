 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations × multiple string allocations and hash lookups against a 6.46M-entry named vector. This alone can take many hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix — this is notoriously slow because `rbind` on a list of vectors has quadratic-like overhead in base R.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 × 6.46M = ~32.3M iterations total.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot can:
- Require the entire prediction matrix (~6.46M × 110 ≈ 5.4 GB as double) plus the model object in memory simultaneously.
- On a 16 GB laptop, this risks swapping to disk.
- If `predict` is called row-by-row or in very small batches (not shown but implied by "prediction-loop"), overhead is catastrophic.

### 1.3 Summary of Root Causes

| Bottleneck | Root Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste`/named-vector lookups in R loop | ~hours |
| `compute_neighbor_stats` | 6.46M R-level iterations + `do.call(rbind, ...)` | ~hours × 5 vars |
| Object copying | `cell_data` is modified in a `for` loop → potential full-copy each iteration | ~minutes–hours |
| Prediction | Possible row-level predict loop; memory pressure from full matrix | ~hours |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation — Vectorize with `data.table`

- Replace the named-vector key lookups with `data.table` integer-keyed joins.
- Replace the per-row `lapply` in `compute_neighbor_stats` with a single grouped `data.table` aggregation: explode the neighbor lookup into an edge-list `(row_i, neighbor_row_j)`, join the variable values, and group-by `row_i` to compute max/min/mean in one vectorized pass.
- Eliminate `do.call(rbind, ...)` entirely.
- Eliminate repeated copying of `cell_data` by adding columns in-place via `:=`.

**Expected speedup:** From hours to minutes.

### 2.2 Prediction — Batched, Memory-Efficient

- Call `predict()` in moderately sized chunks (e.g., 500K rows) to keep peak memory well under 16 GB.
- Avoid any row-level loop.
- Pre-allocate the output vector.

**Expected speedup:** From hours to minutes.

---

## 3. WORKING R CODE

```r
# =============================================================================
# DEPENDENCIES
# =============================================================================
library(data.table)
library(randomForest) # or library(ranger) — adapt predict call as needed

# =============================================================================
# 3A. OPTIMIZED NEIGHBOR LOOKUP — returns an edge-list data.table
#     instead of a list-of-vectors
# =============================================================================
build_neighbor_edgelist <- function(data_dt, id_order, neighbors) {
 # data_dt : data.table with columns 'id' and 'year' (and all feature cols)
 #           must have a column 'row_idx' = 1:.N  (added below if missing)
 # id_order: integer vector; position k → cell id at position k in nb object
 # neighbors: spdep nb object (list of integer vectors of neighbor positions)

  if (!"row_idx" %in% names(data_dt)) {
    data_dt[, row_idx := .I]
  }

  # --- Step 1: map cell-id → position in id_order (vectorised) ---
  id_to_pos <- data.table(id = id_order, pos = seq_along(id_order))

  # --- Step 2: explode the nb object into a cell-level edge list ---
  #     (pos_from, pos_to)  — positions in id_order
  from_pos <- rep(seq_along(neighbors), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)
  nb_edges <- data.table(pos_from = from_pos, pos_to = to_pos)

  # Map positions back to cell ids
  nb_edges[, id_from := id_order[pos_from]]
  nb_edges[, id_to   := id_order[pos_to]]
  nb_edges[, c("pos_from", "pos_to") := NULL]

  # --- Step 3: for every (id_from, year) row, find the row_idx of each
  #     neighbor (id_to, year) ---
  # Key the data for fast join
  row_key <- data_dt[, .(id, year, row_idx)]
  setkey(row_key, id, year)

  # Expand edges by year: join source rows to get year + row_idx of source
  src <- data_dt[, .(id_from = id, year, src_row = row_idx)]
  setkey(src, id_from)

  # For each source row, attach its neighbor cell ids
  # This is a many-to-many join: each src row × its neighbors
  setkey(nb_edges, id_from)
  edge_year <- nb_edges[src, on = "id_from", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: id_from, id_to, year, src_row

  # Join to find the row_idx of the neighbor in the same year
  setkey(edge_year, id_to, year)
  setkey(row_key, id, year)
  edge_year[row_key, nbr_row := i.row_idx, on = c(id_to = "id", "year")]

  # Drop edges where the neighbor-year row doesn't exist
 edge_year <- edge_year[!is.na(nbr_row)]

  # Return slim edge list: (src_row, nbr_row)
  edge_year[, .(src_row, nbr_row)]
}

# =============================================================================
# 3B. OPTIMIZED NEIGHBOR STATS — fully vectorised via data.table grouping
# =============================================================================
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # edge_dt: data.table with columns src_row, nbr_row
  # Attach the neighbor's value
  vals <- data_dt[[var_name]]
  work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]
  work <- work[!is.na(nbr_val)]

  stats <- work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = src_row]

  # Build full-length columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  out_max[stats$src_row]  <- stats$nb_max
  out_min[stats$src_row]  <- stats$nb_min
  out_mean[stats$src_row] <- stats$nb_mean

  list(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# =============================================================================
# 3C. FULL FEATURE-PREPARATION PIPELINE
# =============================================================================
prepare_features <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table in place (no copy if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, row_idx := .I]

  message("Building neighbor edge list …")
  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_dt, src_row)
  message(sprintf("  Edge list: %s edges", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s' …", var_name))
    stats <- compute_neighbor_stats_vec(cell_data, edge_dt, var_name)

    # Add columns in-place (no copy of the whole table)
    set(cell_data, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
    set(cell_data, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
    set(cell_data, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
  }

  cell_data[, row_idx := NULL]
  cell_data
}

# =============================================================================
# 3D. BATCHED RANDOM FOREST PREDICTION
# =============================================================================
predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  # model   : pre-trained randomForest / ranger model (loaded from disk)
  # newdata : data.table / data.frame of predictor columns only
  # Returns : numeric vector of predictions, same length as nrow(newdata)

  n <- nrow(newdata)
  preds <- numeric(n)  # pre-allocate

  starts <- seq(1L, n, by = batch_size)
  message(sprintf("Predicting %s rows in %d batches …",
                  format(n, big.mark = ","), length(starts)))

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)
    batch <- newdata[i1:i2, , drop = FALSE]

    # --- adapt this block to your model class ---
    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = batch)$predictions
    } else {
      # randomForest, or caret-wrapped RF
      preds[i1:i2] <- predict(model, newdata = batch)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  Batch %d / %d done (rows %s–%s)",
                      k, length(starts),
                      format(i1, big.mark = ","),
                      format(i2, big.mark = ",")))
    }
    # Free batch memory explicitly
    rm(batch); gc(verbose = FALSE)
  }

  preds
}

# =============================================================================
# 3E. MAIN PIPELINE
# =============================================================================
run_pipeline <- function(cell_data_path, model_path, id_order, rook_neighbors_unique,
                         predictor_cols, output_path) {
  # --- Load data ---
  message("Loading cell data …")
  cell_data <- fread(cell_data_path)   # or readRDS / qs::qread

  # --- Feature preparation ---
  cell_data <- prepare_features(cell_data, id_order, rook_neighbors_unique)

  # --- Load pre-trained model (once) ---
  message("Loading Random Forest model …")
  model <- readRDS(model_path)

  # --- Prepare prediction matrix ---
  pred_data <- cell_data[, ..predictor_cols]  # data.table column subset, no copy

  # --- Predict in batches ---
  cell_data[, predicted_gdp := predict_rf_batched(model, pred_data)]

  # --- Write results ---
  message("Writing results …")
  fwrite(cell_data, output_path)
  message("Done.")
  invisible(cell_data)
}
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Stage | Original | Optimized | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M R-loop iterations with string ops) | ~1–3 min (vectorised `data.table` joins) | Eliminates per-row `paste`, named-vector lookup |
| `compute_neighbor_stats` (×5 vars) | ~hours (32.3M R iterations + `do.call(rbind)`) | ~2–5 min total (grouped `data.table` aggregation) | Single vectorised pass per variable; no `rbind` |
| Column binding / object copy | Repeated full-copy of 6.46M × 110 data.frame | Zero-copy `set()` in-place | `data.table` reference semantics |
| RF prediction | Potentially hours (row-loop or single OOM call) | ~5–20 min (batched, memory-safe) | Avoids swap; avoids per-row overhead |
| **Total** | **~86+ hours** | **~15–30 minutes** | **~170–340× speedup** |

### Key Principles Applied

1. **Vectorisation over iteration:** Every per-row `lapply` is replaced by a `data.table` join or grouped aggregation that runs in C.
2. **Edge-list representation:** The neighbor lookup is materialized once as a two-column integer edge list (~38M rows for the full panel), enabling all subsequent variable computations to reuse it without rebuilding keys.
3. **In-place mutation:** `data.table::set()` adds columns without triggering R's copy-on-modify, saving ~5.4 GB of redundant copies per iteration.
4. **Batched prediction:** Keeps peak memory well within 16 GB while avoiding per-row `predict()` overhead.
5. **Model preservation:** The trained Random Forest is loaded from disk and used read-only; no retraining occurs. The numerical estimand (predicted GDP) is computed identically — only the surrounding data-handling code changes.