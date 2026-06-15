 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookups, and NA filtering — all inside an interpreted R loop. |
| **`paste` key construction** | Creating ~6.46M × avg-4-neighbors string keys (≈26M strings) is extremely expensive. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | R's named vector lookup is O(n) hash-probe per call; doing it ~26M times is slow. |
| **Output is a list of 6.46M integer vectors** | Massive memory overhead from list structure (~50–100 bytes per list element overhead × 6.46M ≈ 0.5–1 GB just for the container). |

**`compute_neighbor_stats`** is the second bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small integer vector, subsets, removes NAs, and computes three summary stats — all in interpreted R. |
| **Called 5 times** | Once per neighbor source variable, so the full loop runs ~32M R-function invocations. |
| **`do.call(rbind, result)` on 6.46M rows** | Binding 6.46M 3-element vectors into a matrix is itself slow and memory-hungry. |

### B. Random Forest Inference Bottleneck

| Problem | Detail |
|---|---|
| **Single monolithic `predict()` call on 6.46M × 110 columns** | `ranger`/`randomForest` predict must build the full data frame in memory. With 110 numeric columns this is ~5.4 GB for the data alone, plus internal copies. On a 16 GB laptop this causes swapping. |
| **Object copying** | If the model is a `randomForest` object, `predict.randomForest` internally coerces to `data.frame` and may copy. Even `ranger` will allocate a prediction workspace. |
| **No chunking** | A single call means peak memory = model + full dataset + prediction workspace simultaneously. |

### C. Estimated Time Breakdown (86+ hours)

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~25–35% |
| `compute_neighbor_stats` (×5 vars) | ~30–40% |
| RF `predict` (memory pressure / swapping) | ~25–35% |

---

## 2. Optimization Strategy

### Feature Preparation: Vectorize Everything with `data.table`

1. **Replace the string-key lookup with a `data.table` integer join.** Convert the neighbor list into a flat edge-list `data.table` with columns `(row_i, neighbor_row_j)`. Build this once using vectorized operations.
2. **Replace `lapply`-based `compute_neighbor_stats` with a grouped `data.table` aggregation** on the flat edge-list joined to the value column — one vectorized pass per variable.
3. **Eliminate the 6.46M-element list entirely.**

Expected speedup: **50–200×** for feature preparation (minutes instead of tens of hours).

### RF Inference: Chunked Prediction

1. **Predict in chunks** (e.g., 500K rows) to keep peak memory well under 16 GB.
2. **Use `ranger` if possible** (faster C++ predict path); if the model is `randomForest`, still chunk.
3. **Pre-allocate the output vector** and fill in-place.

### Memory

1. Use `data.table` in-place `:=` assignment instead of repeated `cell_data <- cbind(...)` copies.
2. Remove intermediate objects and `gc()` between phases.

---

## 3. Working R Code

```r
# ============================================================
# 0. Dependencies
# ============================================================
library(data.table)
# library(ranger)        # if model is ranger
# library(randomForest)  # if model is randomForest

# ============================================================
# 1. Convert cell_data to data.table (in-place, no copy)
# ============================================================
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure a row index exists (used for joins)
cell_data[, .row_idx := .I]

# ============================================================
# 2. Build flat neighbor edge-list (vectorised)
#    Replaces build_neighbor_lookup entirely
# ============================================================
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors) {
  # --- Map each cell id to its position in id_order ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build a lookup: (id, year) -> row index ---
  key_dt <- cell_dt[, .(id, year, .row_idx)]
  setkey(key_dt, id, year)

  # --- Expand neighbor list into flat (ref_idx, neighbor_cell_id) pairs ---
  #     neighbors[[k]] gives the neighbor positions in id_order for id_order[k]
  n_lengths <- lengths(neighbors)
  from_ref  <- rep(seq_along(neighbors), times = n_lengths)
  to_ref    <- unlist(neighbors, use.names = FALSE)

  from_id <- id_order[from_ref]
  to_id   <- id_order[to_ref]

  edge_dt <- data.table(from_id = from_id, to_id = to_id)

  # --- For every (from_id, year) row, find the row index of (to_id, year) ---
  #     Step 1: attach row indices and years for from_id
  from_info <- cell_dt[, .(from_id = id, year, from_row = .row_idx)]
  edge_year <- edge_dt[from_info, on = "from_id", allow.cartesian = TRUE,
                        nomatch = NULL]
  # edge_year now has: from_id, to_id, year, from_row

  # --- Step 2: join to get to_row (row index of the neighbor in same year) ---
  to_info <- cell_dt[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_info, to_id, year)
  setkey(edge_year, to_id, year)

  edge_year <- to_info[edge_year, nomatch = NA]
  # Keep only matched rows
  edge_year <- edge_year[!is.na(to_row)]

  # Return minimal columns
  edge_year[, .(from_row, to_row)]
}

cat("Building neighbor edge-list …\n")
system.time({
  edge_list <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_list, from_row)
})
# edge_list is ~10-30M rows × 2 integer columns ≈ 200-500 MB


# ============================================================
# 3. Compute & attach neighbor features (vectorised)
#    Replaces compute_neighbor_stats + outer loop
# ============================================================
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Attach the neighbor's value to each edge
  edge_dt[, val := cell_dt[[var_name]][to_row]]

  # Grouped aggregation: max, min, mean per from_row
  stats <- edge_dt[!is.na(val),
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   by = from_row]

  # Column names matching original pipeline convention
  col_max  <- paste0("neighbor_", var_name, "_max")
  col_min  <- paste0("neighbor_", var_name, "_min")
  col_mean <- paste0("neighbor_", var_name, "_mean")

  # In-place assignment (no copy of cell_dt)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]

  cell_dt[stats$from_row, (col_max)  := stats$nb_max]
  cell_dt[stats$from_row, (col_min)  := stats$nb_min]
  cell_dt[stats$from_row, (col_mean) := stats$nb_mean]

  # Clean up temp column on edge_dt
  edge_dt[, val := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features …\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat("  ", var_name, "\n")
    compute_and_add_neighbor_features_fast(cell_data, edge_list, var_name)
  }
})

# Free the edge list
rm(edge_list); gc()


# ============================================================
# 4. Chunked Random Forest Prediction
#    Preserves trained model; preserves original numerical output
# ============================================================
chunked_predict_rf <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  preds <- numeric(n)           # pre-allocate full output vector

  starts <- seq(1L, n, by = chunk_size)
  n_chunks <- length(starts)

  for (ci in seq_along(starts)) {
    i1 <- starts[ci]
    i2 <- min(i1 + chunk_size - 1L, n)
    cat(sprintf("  Predicting chunk %d / %d  (rows %d – %d)\n",
                ci, n_chunks, i1, i2))

    chunk <- newdata[i1:i2, ]

    # --- Works for both ranger and randomForest objects ---
    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = chunk)$predictions
    } else {
      # randomForest
      preds[i1:i2] <- predict(model, newdata = chunk)
    }

    # Hint GC to release chunk workspace
    rm(chunk)
    if (ci %% 5 == 0) gc()
  }

  preds
}

# --- Prepare prediction data frame (drop non-feature columns) ---
# Adjust feature_cols to match whatever the trained model expects.
# If using ranger, feature_cols can be extracted from the model object.
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else {
  # randomForest stores variable names used during training
  feature_cols <- rownames(importance(rf_model))
}

# Subset to feature columns only (avoids sending id/year/target to predict)
pred_data <- cell_data[, ..feature_cols]   # data.table column selection

cat("Running chunked RF prediction …\n")
system.time({
  cell_data[, predicted_gdp := chunked_predict_rf(rf_model, pred_data,
                                                   chunk_size = 500000L)]
})

rm(pred_data); gc()

cat("Done.\n")
```

---

## 4. Expected Impact

| Phase | Before | After | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~20–30 hrs | ~2–5 min | ~300× |
| `compute_neighbor_stats` (×5) | ~30–40 hrs | ~3–8 min | ~300× |
| RF `predict` | ~15–20 hrs (with swap) | ~20–60 min (in-RAM) | ~20× |
| **Total** | **86+ hrs** | **~30–75 min** | **~70–170×** |

### Why Each Change Matters

1. **`build_neighbor_edgelist`**: Replaces 6.46M interpreted R iterations (each with `paste`, character matching) with a single vectorized `data.table` cartesian join. The join uses C-level hash/merge — orders of magnitude faster.

2. **`compute_and_add_neighbor_features_fast`**: Replaces 6.46M `lapply` calls per variable with one `data.table` grouped aggregation (`[, .(max, min, mean), by = from_row]`), which runs in optimized C with radix-sort grouping. In-place `:=` assignment avoids copying the 6.46M-row table.

3. **Chunked prediction**: Keeps peak memory at roughly `model_size + chunk_size × 110 columns` (~400 MB per chunk) instead of `model_size + 6.46M × 110` (~5.4 GB), eliminating OS swap thrashing on a 16 GB laptop. The numerical predictions are identical — chunking only controls how many rows enter `predict()` at once.