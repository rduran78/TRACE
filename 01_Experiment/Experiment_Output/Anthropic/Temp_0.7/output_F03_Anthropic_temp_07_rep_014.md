 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows. For each row it:
- Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) — these are O(n) hash lookups but repeated 6.46M times with string pasting and `paste(..., sep="_")` allocation each iteration.
- Creates intermediate character vectors (`neighbor_keys`) per row.
- Net effect: millions of small allocations, string concatenations, and named-vector lookups. This alone can take **hours**.

**`compute_neighbor_stats`:** Called 5 times (once per source variable). Each call iterates over 6.46M entries in `neighbor_lookup`, subsetting a numeric vector and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M 3-element vectors is extremely slow — `do.call(rbind, list_of_6.46M_vectors)` alone creates a massive temporary list-to-matrix conversion.

**Outer loop:** Calls `compute_and_add_neighbor_features` 5 times, each presumably calling `compute_neighbor_stats`. Each call copies the entire `cell_data` data.frame when assigning new columns (`cell_data <- ...`), triggering R's copy-on-modify semantics on a ~6.46M × 110+ column object.

### 1.2 Random Forest Inference Bottlenecks

- Predicting 6.46M rows × 110 features through a Random Forest (likely `ranger` or `randomForest`) in a single `predict()` call can require **massive memory** (the model object itself + prediction workspace). On 16 GB RAM this can cause swapping.
- If `randomForest::predict.randomForest` is used (rather than `ranger`), it is single-threaded and substantially slower.
- If prediction is done row-by-row or in a naive loop, that compounds the problem enormously.

### 1.3 Memory Pressure

- 6.46M rows × 110 columns × 8 bytes ≈ **5.7 GB** just for the numeric matrix. With R's overhead, copies, and the model in memory, 16 GB is tight. Any unnecessary copy doubles consumption and triggers GC thrashing or swapping.

### Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named-vector lookups | ~hours |
| `compute_neighbor_stats` | `lapply` + `do.call(rbind, ...)` over 6.46M elements, called 5× | ~hours |
| Column assignment in loop | Copy-on-modify of full data.frame 5× | ~tens of minutes + RAM spikes |
| RF prediction | Possibly single-threaded, possibly un-batched, full matrix copy | ~hours |
| Overall memory | Repeated large copies on 16 GB machine | Swapping, GC thrashing |

---

## 2. OPTIMIZATION STRATEGY

### A. Replace data.frame with `data.table` (eliminate copy-on-modify)

`data.table` supports **in-place column addition** via `:=`, eliminating the repeated ~5.7 GB copies.

### B. Vectorize `build_neighbor_lookup` entirely

Instead of building a per-row R list (6.46M entries), build a **flat edge-list** (a two-column integer matrix: `from_row → to_row`) using vectorized operations. This replaces 6.46M `paste` + lookup iterations with a single vectorized join.

### C. Vectorize `compute_neighbor_stats` with `data.table` grouped operations

Use the flat edge-list as a `data.table`, join in the variable values, and compute `max/min/mean` grouped by `from_row` — fully vectorized C-level aggregation.

### D. Batch RF prediction with `ranger`

- If the model is `randomForest`, convert it or re-wrap prediction.
- If `ranger`, use `predict()` with `num.threads` and process in **chunks** (~500K rows) to control peak memory.

### E. Memory discipline

- Remove intermediate objects aggressively (`rm()` + `gc()`).
- Convert prediction input to a matrix once, predict in chunks, write results back.

### Projected speedup

| Component | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~2–4 hrs | ~20–60 sec |
| `compute_neighbor_stats` ×5 | ~3–6 hrs | ~2–5 min total |
| Column assignment ×5 | ~30 min + RAM | ~seconds (in-place) |
| RF prediction | ~1–4 hrs | ~5–20 min |
| **Total** | **~86+ hrs** | **~30–60 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- 0. Convert cell_data to data.table (if not already) --------------------
#    Assumes cell_data is a data.frame/data.table with columns: id, year, ...
#    Assumes rook_neighbors_unique is an nb object (list of integer index vectors)
#    Assumes id_order is the vector of cell IDs corresponding to nb indices
#    Assumes rf_model is the pre-trained Random Forest model (ranger or randomForest)

if (!is.data.table(cell_data)) {
  setDT(cell_data)  # convert in place — no copy
}

# ---- 1. VECTORIZED NEIGHBOR LOOKUP (flat edge-list) -------------------------

build_neighbor_edgelist <- function(dt, id_order, nb_obj) {
  # Map each nb index to its cell id
  # nb_obj[[k]] gives the neighbor indices (in id_order) for cell id_order[k]

  n_cells <- length(id_order)
  stopifnot(length(nb_obj) == n_cells)

  # Build cell-level edge list: from_cell_id -> to_cell_id
  from_cell <- rep(
    id_order,
    times = lengths(nb_obj)
  )
  to_cell <- id_order[unlist(nb_obj, use.names = FALSE)]

  cell_edges <- data.table(from_id = from_cell, to_id = to_cell)

  # Build row-index lookup: (id, year) -> row index in dt
  dt[, .row_idx := .I]

  # For each row in dt, we need to find its neighbors' rows (same year).
  # Strategy: join cell_edges with dt twice — once for "from" rows, once for "to" rows.

  # Unique years
  years <- unique(dt$year)

  # Create a keyed lookup: id, year -> row_idx
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Expand cell_edges across all years (vectorized cross-join)
  # This creates the full (from_row, to_row) edge list.

  # More memory-efficient: iterate by year (28 years — trivial loop)
  edge_list_parts <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Row indices for this year
    yr_lookup <- row_lookup[year == yr]
    setkey(yr_lookup, id)

    # Join from_id to get from_row
    merged <- cell_edges[yr_lookup, on = .(from_id = id), nomatch = 0L,
                         .(from_row = i..row_idx, to_id)]

    # Join to_id to get to_row
    setkey(yr_lookup, id)
    merged <- yr_lookup[merged, on = .(id = to_id), nomatch = 0L,
                        .(from_row, to_row = .row_idx)]

    edge_list_parts[[yi]] <- merged
  }

  edge_dt <- rbindlist(edge_list_parts, use.names = TRUE)

  # Clean up temporary column
  dt[, .row_idx := NULL]

  return(edge_dt)
}

cat("Building vectorized neighbor edge list...\n")
t0 <- proc.time()
edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge list built: %d edges in %.1f sec\n",
            nrow(edge_dt), (proc.time() - t0)[3]))


# ---- 2. VECTORIZED NEIGHBOR STATS -------------------------------------------

compute_and_add_neighbor_features_fast <- function(dt, var_name, edge_dt) {
  # Pull the variable values for the "to" (neighbor) rows
  vals <- dt[[var_name]]
  edge_dt[, nval := vals[to_row]]

  # Drop NAs before aggregation
  valid_edges <- edge_dt[!is.na(nval)]

  # Grouped aggregation — fully vectorized at C level
  stats <- valid_edges[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]

  # Prepare output columns (default NA)
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Assign in-place using data.table := (no copy of dt)
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]

  dt[stats$from_row, (max_col)  := stats$nb_max]
  dt[stats$from_row, (min_col)  := stats$nb_min]
  dt[stats$from_row, (mean_col) := stats$nb_mean]

  # Clean temp column from edge_dt
  edge_dt[, nval := NULL]

  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
t0 <- proc.time()
for (var_name in neighbor_source_vars) {
  t1 <- proc.time()
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  cat(sprintf("  %s done in %.1f sec\n", var_name, (proc.time() - t1)[3]))
}
cat(sprintf("All neighbor features done in %.1f sec\n", (proc.time() - t0)[3]))

# Free the edge list if no longer needed
rm(edge_dt)
gc()


# ---- 3. BATCHED RANDOM FOREST PREDICTION ------------------------------------

predict_rf_batched <- function(model, dt, feature_cols, batch_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)

  # Determine if model is ranger or randomForest
  is_ranger <- inherits(model, "ranger")

  # Pre-select feature columns as a matrix (or data.frame for randomForest)
  # Process in chunks to control memory
  n_batches <- ceiling(n / batch_size)

  cat(sprintf("Predicting %d rows in %d batches (batch_size=%d)...\n",
              n, n_batches, batch_size))

  for (b in seq_len(n_batches)) {
    idx_start <- (b - 1L) * batch_size + 1L
    idx_end   <- min(b * batch_size, n)
    idx       <- idx_start:idx_end

    # Extract batch — data.table subsetting is efficient
    batch_df <- dt[idx, ..feature_cols]

    if (is_ranger) {
      # ranger::predict returns a list with $predictions
      batch_pred <- predict(model, data = batch_df,
                            num.threads = parallel::detectCores(logical = FALSE))$predictions
    } else {
      # randomForest::predict
      batch_pred <- predict(model, newdata = batch_df)
    }

    preds[idx] <- batch_pred

    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("  Batch %d/%d complete (rows %d-%d)\n",
                  b, n_batches, idx_start, idx_end))
    }

    # Free batch memory
    rm(batch_df, batch_pred)
    if (b %% 10 == 0) gc()
  }

  return(preds)
}

# Identify feature columns (all predictors used by the model)
# Adjust this to match your actual feature column names:
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names in the model
  feature_cols <- attr(rf_model$terms, "term.labels")
  if (is.null(feature_cols)) {
    # If trained with x/y interface, rownames of importance
    feature_cols <- rownames(rf_model$importance)
  }
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features exist
missing_cols <- setdiff(feature_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop("Missing feature columns in cell_data: ",
       paste(missing_cols, collapse = ", "))
}

cat("Starting Random Forest prediction...\n")
t0 <- proc.time()
cell_data[, predicted_gdp := predict_rf_batched(rf_model, cell_data,
                                                 feature_cols,
                                                 batch_size = 500000L)]
cat(sprintf("Prediction complete in %.1f sec\n", (proc.time() - t0)[3]))


# ---- 4. FINAL OUTPUT --------------------------------------------------------
# cell_data now has the column `predicted_gdp` with the original numerical
# estimand preserved (same RF model, same features, same predictions).

cat(sprintf("Done. Output: %d rows, %d columns\n", nrow(cell_data), ncol(cell_data)))
cat(sprintf("Prediction range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **`setDT()` in place** | Zero-copy conversion; all subsequent `:=` assignments are in-place (no 5.7 GB copies). |
| **Flat edge-list instead of per-row list** | Eliminates 6.46M `paste`/string-lookup iterations. The 28-year loop (trivial overhead) keeps peak memory bounded. |
| **`data.table` grouped aggregation** | `max`/`min`/`mean` by group uses data.table's radix-sort + C-level aggregation — orders of magnitude faster than `lapply` + `do.call(rbind, ...)`. |
| **Batched prediction (500K rows)** | Keeps peak memory for the prediction matrix at ~500K × 110 × 8 bytes ≈ 440 MB per batch, well within 16 GB even with the model loaded. |
| **`num.threads` for ranger** | Exploits all physical cores for tree-level parallelism during inference. |
| **Model is never retrained** | Only `predict()` is called; the model object is read-only throughout. |
| **Numerical estimand preserved** | Same model + same features + same `predict()` call = identical numerical output. |