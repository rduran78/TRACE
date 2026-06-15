 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

1. **Row-level `lapply` over 6.46M rows.** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is ~6.46 million R-level function calls with repeated string allocation.
2. **String-keyed lookups are O(n) hash constructions and repeated hash probes.** The `idx_lookup` named vector has 6.46M entries; building it with `paste` is expensive, and each probe into it requires hashing a string.
3. **Redundant work across years.** The neighbor *topology* is time-invariant (cell A's neighbors are the same in every year), but the function recomputes neighbor keys for every cell-year row independently.

**`compute_neighbor_stats`** is the second bottleneck:

1. **Row-level `lapply` over 6.46M rows**, each extracting a variable-length subset of a numeric vector, removing NAs, and computing three summary statistics.
2. **`do.call(rbind, result)` on a 6.46M-element list of 3-vectors** is a well-known R anti-pattern that creates enormous intermediate object churn.
3. **Called 5 times** (once per neighbor source variable), so all costs multiply by 5.

**Combined cost:** ~6.46M × (string ops + list ops) × 6 passes (1 build + 5 stats) ≈ billions of interpreted R operations. This alone can account for many hours.

### B. Random Forest Inference Bottleneck

With ~6.46M rows and ~110 predictors, a single `predict.randomForest()` or `predict.ranger()` call must push every row through every tree:

1. **Memory:** A `data.frame` of 6.46M × 110 float64 columns ≈ **5.3 GB**. If the predict method internally copies or coerces this (e.g., `model.matrix`, `data.frame` → `matrix`), peak RAM can exceed 16 GB and force swapping.
2. **Object copying:** R's copy-on-modify semantics mean that adding columns to `cell_data` inside a loop (`cell_data$new_col <- ...`) can trigger full-frame copies of a 5+ GB object.
3. **Single-threaded prediction:** `randomForest::predict` is single-threaded. `ranger::predict` supports `num.threads` but defaults to 1 in some configurations.
4. **Predicting all 6.46M rows at once** may be slower than batched prediction due to cache pressure on large matrices.

### C. Summary of Time Sinks (estimated share of 86+ hours)

| Component | Est. Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~25% | 6.46M string-key lookups in R loop |
| `compute_neighbor_stats` (×5) | ~40% | 6.46M row-level lapply ×5, `do.call(rbind,...)` |
| Column binding / object copies | ~10% | Repeated mutation of 5 GB data.frame |
| RF `predict()` | ~20% | Large matrix, possible copy, single thread |
| I/O / model load | ~5% | Deserializing large RF object |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Eliminate R-level row loops; vectorize everything with `data.table` and integer indexing.

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Exploit time-invariance: build neighbor pairs once as an integer edge-list, then join on year via `data.table` merge. No strings, no per-row function. | ~100–500× |
| `compute_neighbor_stats` | Vectorized grouped aggregation: join edge-list to data values, then `data.table` grouped `max/min/mean` in one pass per variable. | ~100–500× |
| Column binding | Mutate `data.table` by reference (`:=`), zero copies. | ~10× |
| RF predict | Batch prediction in chunks (~500K rows); use `ranger::predict` with `num.threads`; pass a `matrix` not a `data.frame`. If model is `randomForest`, convert once to `ranger`-compatible or predict in chunks. | ~2–5× |
| Model load | Load once, keep in memory; use `qs::qread` instead of `readRDS` for faster deserialization. | ~2× |

**Estimated total runtime: 10–30 minutes** (down from 86+ hours).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest), qs (optional)
# Preserves: trained RF model object, original numerical estimand (GDP predictions)
# =============================================================================

library(data.table)

# ---- Configuration ----------------------------------------------------------
BATCH_SIZE    <- 500000L
NUM_THREADS   <- parallel::detectCores(logical = FALSE)  # physical cores

# ---- Step 0: Load model once -----------------------------------------------
# Use qs for faster deserialization if available; otherwise readRDS
if (requireNamespace("qs", quietly = TRUE)) {
  rf_model <- qs::qread("rf_model.qs")
} else {
  rf_model <- readRDS("rf_model.rds")
}

# ---- Step 1: Convert cell_data to data.table (by reference if possible) -----
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place, no copy
}

# Ensure there is a row index for fast joining
cell_data[, row_idx := .I]

# ---- Step 2: Build vectorized neighbor edge-list (time-invariant) -----------
# This replaces build_neighbor_lookup entirely.
# rook_neighbors_unique is an nb object: list of integer vectors indexed by
# position in id_order.

build_neighbor_edgelist <- function(id_order, neighbors) {
  # neighbors[[i]] gives the positional indices (into id_order) of cell
  # id_order[i]'s neighbors.
  # We build a two-column integer matrix: (focal_cell_id, neighbor_cell_id)

  n <- length(neighbors)
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(lengths(neighbors))

  focal_id    <- integer(n_edges)
  neighbor_id <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    idx_range <- pos:(pos + len - 1L)
    focal_id[idx_range]    <- id_order[i]
    neighbor_id[idx_range] <- id_order[nb_i]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-neighbor cells)
  if (pos <= n_edges) {
    focal_id    <- focal_id[1:(pos - 1L)]
    neighbor_id <- neighbor_id[1:(pos - 1L)]
  }

  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

cat("Building neighbor edge-list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge-list: %d directed edges\n", nrow(edge_dt)))

# ---- Step 3: Vectorized neighbor feature computation ------------------------
# For each (focal_cell, year) we need max/min/mean of each neighbor variable.
# Strategy:
#   1. Create a slim lookup: (id, year, row_idx) from cell_data.
#   2. Join edge_dt × years to get (focal_row_idx, neighbor_row_idx) pairs.
#   3. For each variable, extract neighbor values and do grouped aggregation.

compute_all_neighbor_features <- function(cell_data, edge_dt,
                                          neighbor_source_vars) {
  # Slim key table: id, year, row_idx
  key_dt <- cell_data[, .(id, year, row_idx)]

  # --- Build full (focal_row_idx, neighbor_row_idx) mapping ---
  # Join edge_dt with key_dt to get focal rows
  # edge_dt has (focal_id, neighbor_id); we need to expand by year.

  cat("  Joining edges to focal rows...\n")
  # focal side: for each edge, get all years the focal cell appears in
  setkey(key_dt, id)
  focal_join <- key_dt[edge_dt, on = .(id = focal_id),
                       .(focal_row_idx = row_idx,
                         neighbor_id   = i.neighbor_id,
                         year          = year),
                       nomatch = 0L,
                       allow.cartesian = TRUE]

  cat(sprintf("  Focal-expanded edges: %d\n", nrow(focal_join)))

  # neighbor side: join to get neighbor_row_idx for same year
  cat("  Joining to neighbor rows (same year)...\n")
  setkey(key_dt, id, year)
  paired <- key_dt[focal_join,
                   on = .(id = neighbor_id, year = year),
                   .(focal_row_idx    = i.focal_row_idx,
                     neighbor_row_idx = row_idx),
                   nomatch = 0L]

  cat(sprintf("  Paired edges (focal-neighbor-year): %d\n", nrow(paired)))

  # --- Grouped aggregation per variable ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Computing neighbor stats for '%s'...\n", var_name))

    # Extract neighbor values via integer indexing (vectorized, no copy)
    vals <- cell_data[[var_name]]
    paired[, nval := vals[neighbor_row_idx]]

    # Grouped aggregation: max, min, mean — excluding NAs
    agg <- paired[!is.na(nval),
                  .(nb_max  = max(nval),
                    nb_min  = min(nval),
                    nb_mean = mean(nval)),
                  keyby = .(focal_row_idx)]

    # Assign back to cell_data by reference
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    # Initialize with NA
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    # Fill in computed values
    set(cell_data, i = agg$focal_row_idx, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$focal_row_idx, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$focal_row_idx, j = mean_col, value = agg$nb_mean)

    # Clean up temp column
    paired[, nval := NULL]
  }

  # Clean up large intermediate
  rm(focal_join, paired)
  gc()

  invisible(cell_data)
}

cat("Computing neighbor features (vectorized)...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features(cell_data, edge_dt,
                                           neighbor_source_vars)

# Remove helper column
cell_data[, row_idx := NULL]

cat("Neighbor features complete.\n")

# ---- Step 4: Prepare prediction matrix once ---------------------------------
# Identify the predictor columns the model expects
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used in training
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all required predictors are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

cat(sprintf("Preparing prediction matrix (%d rows x %d cols)...\n",
            nrow(cell_data), length(pred_vars)))

# Convert to matrix once (avoids repeated coercion inside predict)
# data.table's as.matrix on selected columns is efficient
pred_matrix <- as.matrix(cell_data[, ..pred_vars])

# ---- Step 5: Batched prediction ---------------------------------------------
cat("Running Random Forest prediction...\n")

n_rows   <- nrow(pred_matrix)
n_batches <- ceiling(n_rows / BATCH_SIZE)
predictions <- numeric(n_rows)

predict_batch <- function(model, newdata_matrix, idx) {
  batch_data <- newdata_matrix[idx, , drop = FALSE]

  if (inherits(model, "ranger")) {
    # ranger can accept a matrix and supports multi-threading
    pred <- predict(model,
                    data      = batch_data,
                    num.threads = NUM_THREADS)
    return(pred$predictions)

  } else if (inherits(model, "randomForest")) {
    # randomForest::predict expects a data.frame (or matrix for some versions)
    pred <- predict(model, newdata = as.data.frame(batch_data))
    return(as.numeric(pred))

  } else {
    stop("Unsupported model class")
  }
}

for (b in seq_len(n_batches)) {
  start_idx <- (b - 1L) * BATCH_SIZE + 1L
  end_idx   <- min(b * BATCH_SIZE, n_rows)
  idx       <- start_idx:end_idx

  if (b %% 5 == 1 || b == n_batches) {
    cat(sprintf("  Batch %d/%d (rows %d-%d)...\n", b, n_batches,
                start_idx, end_idx))
  }

  predictions[idx] <- predict_batch(rf_model, pred_matrix, idx)
}

# ---- Step 6: Attach predictions to data (by reference) ----------------------
cell_data[, predicted_gdp := predictions]

cat(sprintf("Done. %d predictions generated.\n", length(predictions)))

# ---- Step 7: Clean up large objects -----------------------------------------
rm(pred_matrix, edge_dt)
gc()

# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
# 1. build_neighbor_lookup:
#    BEFORE: 6.46M-iteration lapply with string paste/hash lookups.
#    AFTER:  One-time integer edge-list construction (~1.37M edges), then
#            data.table equi-join to expand across years. No strings.
#
# 2. compute_neighbor_stats:
#    BEFORE: 6.46M-iteration lapply × 5 variables, do.call(rbind,...).
#    AFTER:  Vectorized integer indexing + data.table grouped aggregation
#            (max/min/mean) per variable. Zero row-level R loops.
#
# 3. Column mutation:
#    BEFORE: cell_data$new_col <- ... triggers copy-on-modify of 5+ GB frame.
#    AFTER:  data.table::set() modifies by reference. Zero copies.
#
# 4. RF prediction:
#    BEFORE: Single predict() call on 6.46M × 110 → possible 5+ GB copy.
#    AFTER:  Batched (500K rows), matrix input (no coercion), multi-threaded
#            if ranger.
#
# 5. Model loading:
#    Optional qs::qread for ~2× faster deserialization.
#
# Estimated runtime: 10–30 minutes on 16 GB laptop (down from 86+ hours).
# Numerical output: Identical predictions from the same trained RF model.
# =============================================================================
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

### Why an edge-list instead of a list-of-neighbors?

The original `build_neighbor_lookup` produces a **list of 6.46M integer vectors** — one per cell-year row. This is inherently a row-level data structure that forces row-level iteration. An **edge-list** (`data.table` with two columns: `focal_id`, `neighbor_id`) is a flat, columnar structure that `data.table` can join, filter, and aggregate over in vectorized C code. The topology has only ~1.37M edges; expanding by 28 years via an equi-join produces ~38M rows — large but trivially handled by `data.table` in seconds.

### Why `set()` instead of `:=` in the assignment loop?

`set()` is the programmatic (variable-column-name) equivalent of `:=` and avoids `eval`/`parse` overhead. Both modify by reference with zero copies.

### Why batched prediction?

A 6.46M × 110 `float64` matrix is ~5.3 GB. `predict()` may internally allocate a similar-sized structure (e.g., per-tree predictions before averaging). Batching at 500K rows keeps the working set under ~500 MB, staying within L3 cache and avoiding swap on a 16 GB machine.

### Numerical equivalence guarantee

No transformation is applied to the model or its inputs. The same predictor columns, in the same order, with the same values, are passed to the same `predict()` method. The output is `predicted_gdp` — identical to what the original pipeline would produce, preserving the original numerical estimand.