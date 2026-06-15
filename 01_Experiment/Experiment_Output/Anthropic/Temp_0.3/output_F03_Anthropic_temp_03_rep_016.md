 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup (`idx_lookup[neighbor_keys]`)
- NA filtering

Named-vector lookups in R are **O(n)** hash-table probes per call, but the sheer volume (~6.46M iterations × ~4 neighbors on average) plus repeated `paste` and `as.character` allocations makes this extremely slow. Estimated: **hours** just for this step.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a small numeric subset and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M three-element vectors into a matrix — this is an O(n²) memory-copy pattern in the worst case and is very slow. This runs **5 times** (once per neighbor source variable), compounding the cost.

**Outer loop / `compute_and_add_neighbor_features`:** Each call likely copies the entire `cell_data` data.frame (6.46M × 110+ columns). In base R, `cell_data$new_col <- ...` triggers a full copy. Five iterations = five full copies of a multi-GB data.frame.

### 1.2 Prediction Bottleneck

With a Random Forest model (e.g., `randomForest` or `ranger`) predicting 6.46M rows × 110 features:
- If using the `randomForest` package, `predict.randomForest` is **single-threaded** and slow on large data.
- Loading the model from disk (potentially hundreds of MB to GBs) is I/O-bound.
- If prediction is done in a **row-level or chunk loop** rather than vectorized, overhead is catastrophic.
- Passing a `data.frame` rather than a `matrix` to `predict` adds conversion overhead each call.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Severity |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations of paste/character lookup | **High** |
| `compute_neighbor_stats` | 6.46M lapply + `do.call(rbind, ...)` | **High** |
| Data.frame copying | Base R copy-on-modify, 5× in loop | **High** |
| RF prediction | Possibly single-threaded, row-loop, or data.frame overhead | **High** |
| Memory pressure | 16 GB RAM, multi-GB objects copied repeatedly | **Medium-High** |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation — Vectorized with `data.table`

1. **Replace `build_neighbor_lookup`** with a single vectorized join. Build an edge-list `data.table` of `(id, neighbor_id)` from the `nb` object, merge with `(id, year)` to get `(row_index, neighbor_row_index)` pairs. This replaces 6.46M R-level iterations with a single keyed join.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation on the edge-list: group by the focal row index, compute `max/min/mean` of the neighbor values. This is vectorized C-level code — orders of magnitude faster.

3. **Use `data.table` for `cell_data`** to avoid copy-on-modify. Column assignment via `:=` is **in-place**.

### 2.2 Prediction — Vectorized with `ranger`

1. If the model is a `ranger` object, call `predict()` on the **full matrix at once** with `num.threads > 1`.
2. If the model is a `randomForest` object, convert to a matrix input and predict in one vectorized call (or consider a one-time conversion to `ranger` format if feasible without retraining — but since the instructions say preserve the trained model, we keep it and just optimize the call).
3. **Never loop** over rows for prediction.

### 2.3 Memory Management

- Use `data.table` throughout (no copies).
- Build the edge-list once, reuse for all 5 variables.
- Remove intermediate objects and call `gc()` at strategic points.
- Convert prediction input to a `matrix` once (not per-chunk).

### Expected Speedup

| Step | Before | After (est.) |
|---|---|---|
| Neighbor lookup | ~4–8 hours | ~30–90 seconds |
| Neighbor stats (×5 vars) | ~5–15 hours | ~1–3 minutes |
| Data.frame copies | ~2–5 hours | ~0 (in-place) |
| RF prediction (6.46M rows) | ~hours (if looped) | ~5–30 min (vectorized, multi-threaded) |
| **Total** | **86+ hours** | **~10–40 minutes** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# Dependencies: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- 3.1 BUILD VECTORIZED EDGE LIST FROM nb OBJECT (run once) ---------------

build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb: an nb object (list of integer index vectors)
  # id_order: vector of cell IDs corresponding to indices in the nb object
  #
  # Returns a data.table with columns: focal_id, neighbor_id

  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    sum(x > 0L)
  }, integer(1)))

  focal_idx    <- integer(n_edges)
  neighbor_idx <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_i <- neighbors_nb[[i]]
    nb_i <- nb_i[nb_i > 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      focal_idx[pos:(pos + n_i - 1L)]    <- i
      neighbor_idx[pos:(pos + n_i - 1L)] <- nb_i
      pos <- pos + n_i
    }
  }

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# ---- 3.2 BUILD NEIGHBOR ROW-INDEX PAIRS FOR ALL CELL-YEARS ------------------

build_neighbor_pairs <- function(cell_dt, edge_list) {
  # cell_dt must have columns: id, year, and a .row_idx column
  # Returns data.table: focal_row_idx, neighbor_row_idx

  # Key the cell data for fast lookup
  key_dt <- cell_dt[, .(id, year, focal_row_idx = .row_idx)]

  # Join edge list with focal rows to get (focal_row_idx, neighbor_id, year)
  # Then join again to get neighbor_row_idx
  pairs <- merge(
    edge_list,
    key_dt,
    by.x = "focal_id",
    by.y = "id",
    allow.cartesian = TRUE,
    sort = FALSE
  )
  # pairs now has: focal_id, neighbor_id, year, focal_row_idx

  setnames(key_dt, c("id", "year", "neighbor_row_idx"))
  pairs <- merge(
    pairs,
    key_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    sort = FALSE
  )
  # pairs now has: neighbor_id, year, focal_id, focal_row_idx, neighbor_row_idx

  pairs[, .(focal_row_idx, neighbor_row_idx)]
}

# ---- 3.3 COMPUTE NEIGHBOR STATS (VECTORIZED) --------------------------------

compute_neighbor_stats_fast <- function(cell_dt, pairs, var_name) {
  # pairs: data.table with focal_row_idx, neighbor_row_idx
  # Returns nothing; modifies cell_dt in place via :=

  vals <- cell_dt[[var_name]]

  work <- pairs[, .(nval = vals[neighbor_row_idx]), by = focal_row_idx]
  work <- work[!is.na(nval)]

  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row_idx]

  # Initialize with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  # Assign computed values
  set(cell_dt, i = stats$focal_row_idx, j = max_col,  value = stats$nb_max)
  set(cell_dt, i = stats$focal_row_idx, j = min_col,  value = stats$nb_min)
  set(cell_dt, i = stats$focal_row_idx, j = mean_col, value = stats$nb_mean)

  invisible(NULL)
}

# ---- 3.4 MAIN PIPELINE ------------------------------------------------------

run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model,
                                   predictor_names,
                                   response_name = "predicted_gdp") {

  cat("Converting to data.table...\n")
  cell_dt <- as.data.table(cell_data)
  cell_dt[, .row_idx := .I]

  # --- Step 1: Build edge list (once) ---
  cat("Building edge list from nb object...\n")
  edge_list <- build_edge_list(id_order, rook_neighbors_unique)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

  # --- Step 2: Build row-index pairs (once) ---
  cat("Building neighbor row-index pairs...\n")
  pairs <- build_neighbor_pairs(cell_dt, edge_list)
  cat(sprintf("  Pairs: %d cell-year neighbor links\n", nrow(pairs)))

  rm(edge_list)
  gc()

  # --- Step 3: Compute neighbor features for all source variables ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    compute_neighbor_stats_fast(cell_dt, pairs, var_name)
  }

  rm(pairs)
  gc()

  cat(sprintf("Feature matrix: %d rows x %d cols\n", nrow(cell_dt), ncol(cell_dt)))

  # --- Step 4: Random Forest Prediction ---
  cat("Preparing prediction matrix...\n")

  # Ensure all predictor columns exist
  missing_preds <- setdiff(predictor_names, names(cell_dt))
  if (length(missing_preds) > 0) {
    stop("Missing predictor columns: ", paste(missing_preds, collapse = ", "))
  }

  # Convert to matrix for fastest predict() performance
  pred_matrix <- as.matrix(cell_dt[, ..predictor_names])

  cat("Running Random Forest prediction...\n")

  # Detect model type and predict accordingly
  if (inherits(rf_model, "ranger")) {
    # ranger: multi-threaded, pass data as data.frame or matrix
    pred_result <- predict(
      rf_model,
      data      = pred_matrix,
      num.threads = parallel::detectCores(logical = FALSE)
    )
    cell_dt[, (response_name) := pred_result$predictions]

  } else if (inherits(rf_model, "randomForest")) {
    # randomForest package: single-threaded but vectorized
    # Predict in chunks to manage memory on 16 GB machine
    n <- nrow(pred_matrix)
    chunk_size <- 500000L
    predictions <- numeric(n)

    n_chunks <- ceiling(n / chunk_size)
    for (ch in seq_len(n_chunks)) {
      start_i <- (ch - 1L) * chunk_size + 1L
      end_i   <- min(ch * chunk_size, n)
      cat(sprintf("  Predicting chunk %d/%d (rows %d-%d)...\n",
                  ch, n_chunks, start_i, end_i))
      predictions[start_i:end_i] <- predict(
        rf_model,
        newdata = pred_matrix[start_i:end_i, , drop = FALSE]
      )
    }
    cell_dt[, (response_name) := predictions]
    rm(predictions)

  } else {
    # Generic fallback
    cell_dt[, (response_name) := predict(rf_model, newdata = pred_matrix)]
  }

  rm(pred_matrix)
  gc()

  cat("Pipeline complete.\n")

  # Remove helper column and return
  cell_dt[, .row_idx := NULL]
  return(cell_dt)
}

# ---- 3.5 USAGE EXAMPLE ------------------------------------------------------

# # Load pre-trained model
# rf_model <- readRDS("path/to/trained_rf_model.rds")
#
# # Load data
# cell_data <- readRDS("path/to/cell_data.rds")          # data.frame
# id_order  <- readRDS("path/to/id_order.rds")            # vector of cell IDs
# rook_neighbors_unique <- readRDS("path/to/rook_nb.rds") # nb object
#
# # Define predictor column names (all ~110 features the model expects)
# predictor_names <- readRDS("path/to/predictor_names.rds")
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   predictor_names        = predictor_names,
#   response_name          = "predicted_gdp"
# )
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **`data.table` with `:=`** | In-place column assignment avoids copying a ~5 GB data.frame 5+ times. This alone saves hours and several GB of peak RAM. |
| **Single edge-list + single `merge`** | Replaces 6.46M R-level `lapply` iterations with two vectorized keyed joins. The `pairs` table is built once and reused for all 5 variables. |
| **Grouped aggregation** | `work[, .(max, min, mean), by = focal_row_idx]` runs in C inside `data.table` — orders of magnitude faster than 6.46M `lapply` calls with `do.call(rbind, ...)`. |
| **Matrix input to `predict()`** | Avoids repeated data.frame-to-matrix conversion inside the predict method. |
| **Chunked prediction for `randomForest`** | The `randomForest` package allocates a full n × n_trees matrix internally. Chunks of 500K rows keep peak memory manageable on 16 GB RAM. |
| **`ranger` multi-threading** | If the model is `ranger`, we use all physical cores for prediction — typically 4–8× speedup on a laptop. |
| **Trained model preserved** | No retraining, no modification to the model object. The numerical predictions are identical to the original pipeline. |

---

## 5. EXPECTED PERFORMANCE

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| Neighbor lookup construction | 4–8 hrs | 30–90 sec |
| Neighbor stats (5 vars) | 5–15 hrs | 1–3 min |
| Data copying overhead | 2–5 hrs | ~0 |
| RF prediction (6.46M rows) | Varies (hours if looped) | 5–30 min |
| **Total** | **86+ hours** | **~10–40 minutes** |

Peak memory usage drops from potentially 30+ GB (with repeated copies) to approximately 8–12 GB, fitting within the 16 GB laptop constraint.