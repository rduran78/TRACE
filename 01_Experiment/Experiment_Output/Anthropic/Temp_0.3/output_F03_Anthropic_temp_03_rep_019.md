 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - `as.character()` conversion and named-vector lookup (`id_to_ref`).
  - Subsetting `neighbors[[ref_idx]]` (an `nb` list element).
  - `paste()` to build composite keys.
  - Named-vector lookup into `idx_lookup` (a named character vector of length 6.46M — every lookup is **O(n) hash probe on a massive vector**).
  - `is.na` filtering and `as.integer` coercion.
- The result is a **list of 6.46 million integer vectors**, consuming enormous memory and taking hours to build because of per-row string operations and hash lookups on a 6.46M-entry named vector.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M elements, each extracting neighbor values, removing NAs, and computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3 million R-level function calls total.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself slow (creates a huge temporary matrix row-by-row).

**Memory pressure**: The neighbor lookup list alone (6.46M entries, each a small integer vector) plus repeated `data.frame` column-binding (`compute_and_add_neighbor_features` likely does `cbind` or `$<-` on a 6.46M-row data.frame) causes repeated full-copy of the data.frame (R's copy-on-modify semantics). With ~110 columns × 6.46M rows × 8 bytes ≈ 5.7 GB, each copy can exhaust 16 GB RAM and trigger swapping.

### B. Random Forest Inference Bottlenecks

- Calling `predict()` on a single large Random Forest model over 6.46M rows with 110 features is memory-intensive: `ranger` or `randomForest` must build the full prediction matrix and traverse every tree.
- If the model is a `randomForest` object (not `ranger`), prediction is single-threaded and slow.
- If prediction is done inside a loop (e.g., year-by-year or chunk-by-chunk) with repeated `predict()` calls, each call has overhead from data validation, factor-level checking, and matrix construction.
- Passing a `data.frame` to `predict()` instead of a `matrix` adds coercion overhead on every call.

### C. Summary of Root Causes

| Root Cause | Impact |
|---|---|
| Per-row `paste`/string-key lookups in `build_neighbor_lookup` | ~hours on 6.46M rows |
| Named-vector lookup on 6.46M-entry vector (hash table) | Slow per-probe |
| `lapply` + R-level loops for neighbor stats (×5 vars) | ~hours |
| `do.call(rbind, ...)` on millions of list elements | Slow matrix assembly |
| Copy-on-modify of large data.frame when adding columns | RAM exhaustion, swapping |
| Possible single-threaded RF prediction (`randomForest::predict`) | Hours for 6.46M rows |
| Possible row-by-row or small-batch prediction loop | Overhead per call |

---

## 2. Optimization Strategy

### Feature Preparation: Vectorize Everything with `data.table`

1. **Replace the neighbor lookup list** with a flat `data.table` join. Instead of building a 6.46M-element list, create a long-form neighbor-pair table (`cell_row → neighbor_row`) using vectorized integer arithmetic. Avoid all `paste()`/string keys — use integer-keyed joins on `(id, year)`.

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation: join the neighbor-pair table to the variable column, then `[, .(max, min, mean), by = cell_row]`. This replaces 6.46M R-level function calls with a single vectorized C-level operation.

3. **Use `data.table` in-place column assignment (`:=`)** to add the 15 new columns (5 vars × 3 stats) without copying the entire table.

### Random Forest Inference: Batch Prediction with Matrix Input

4. **Convert to matrix once** before prediction. Avoid repeated `data.frame` → `matrix` coercion.

5. **If the model is `randomForest`**: convert it to `ranger` format using the same trees (not possible directly), OR chunk the prediction into blocks of ~500K rows to control peak memory, OR simply accept single-threaded prediction but ensure the input is a pre-built numeric matrix.

6. **If the model is `ranger`**: use `num.threads` to parallelize prediction.

7. **Predict in moderately large chunks** (~500K–1M rows) to stay within RAM while minimizing per-call overhead.

### Memory Management

8. Target peak RAM ≈ 8–10 GB (within 16 GB). The `data.table` approach avoids copies. Chunked prediction avoids materializing all tree outputs simultaneously.

### Expected Speedup

| Stage | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~20–40 hrs | ~1–3 min |
| `compute_neighbor_stats` (×5) | ~30–40 hrs | ~2–5 min |
| Column binding / copies | ~hours (swapping) | ~seconds (`:=`) |
| RF prediction (6.46M rows) | ~1–6 hrs | ~10–30 min |
| **Total** | **86+ hrs** | **~15–40 min** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# =============================================================================
# Requirements: data.table, ranger (or randomForest), Matrix (optional)
# Preserves: trained RF model object, original numerical estimand (GDP)
# =============================================================================

library(data.table)

# ---- STEP 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year are integer for fast keyed joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Add a row index (will be used as the primary key for aggregation)
cell_data[, .row_idx := .I]


# ---- STEP 1: Build flat neighbor-pair table (vectorized, no strings) --------
# rook_neighbors_unique is an nb object: a list of length = # cells,
# where element i contains integer indices of neighbors of cell i
# id_order is the vector mapping position -> cell id

build_neighbor_pairs_dt <- function(cell_data, id_order, neighbors) {
  # id_order[i] is the cell id at position i in the nb object
  # neighbors[[i]] contains integer positions of neighbors of cell at position i

  n_cells <- length(id_order)

  # Map: cell_id -> position in nb object
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)

  # Build flat edge list: (focal_cell_id, neighbor_cell_id)
  # Vectorized expansion of the nb list
  n_neighbors <- lengths(neighbors)  # integer vector, length = n_cells
  focal_pos   <- rep.int(seq_len(n_cells), n_neighbors)
  nbr_pos     <- unlist(neighbors, use.names = FALSE)

  focal_ids <- id_order[focal_pos]
  nbr_ids   <- id_order[nbr_pos]

  # Create edge table (cell-level, year-independent)
  edge_dt <- data.table(
    focal_id = as.integer(focal_ids),
    nbr_id   = as.integer(nbr_ids)
  )
  rm(focal_pos, nbr_pos, focal_ids, nbr_ids)

  # Now cross-join with years to get (focal_id, year, nbr_id, year) pairs

# But that would be 1.37M edges × 28 years = 38.5M rows — manageable.
  # Instead, join edges to cell_data rows for both focal and neighbor.

  # Key cell_data for fast join
  setkey(cell_data, id, year)

  # Get unique years
  years <- sort(unique(cell_data$year))

  # Expand edges across all years
  # CJ-like expansion: each edge exists for every year
  year_dt <- data.table(year = as.integer(years))
  edge_year_dt <- edge_dt[, CJ_wrapper := TRUE][
    year_dt, on = "CJ_wrapper", allow.cartesian = TRUE
  ]
  # Cleaner approach:
  edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year_dt[, focal_id := edge_dt$focal_id[edge_idx]]
  edge_year_dt[, nbr_id   := edge_dt$nbr_id[edge_idx]]
  edge_year_dt[, edge_idx := NULL]

  # Join to get focal row index
  focal_map <- cell_data[, .(id, year, focal_row = .row_idx)]
  setkey(focal_map, id, year)
  edge_year_dt <- focal_map[edge_year_dt, on = .(id = focal_id, year = year),
                            nomatch = NULL]
  setnames(edge_year_dt, "id", "focal_id")

  # Join to get neighbor row index
  nbr_map <- cell_data[, .(id, year, nbr_row = .row_idx)]
  setkey(nbr_map, id, year)
  edge_year_dt <- nbr_map[edge_year_dt, on = .(id = nbr_id, year = year),
                           nomatch = NULL]
  setnames(edge_year_dt, "id", "nbr_id")

  # Result: data.table with columns (focal_row, nbr_row) — all we need

  edge_year_dt[, .(focal_row, nbr_row)]
}

# --- More memory-efficient version (avoids 38M-row CJ) ---
build_neighbor_pairs_dt <- function(cell_data, id_order, neighbors) {

  n_cells <- length(id_order)

  # Flat edge list at cell level
  n_nbrs    <- lengths(neighbors)
  focal_pos <- rep.int(seq_len(n_cells), n_nbrs)
  nbr_pos   <- unlist(neighbors, use.names = FALSE)

  focal_ids <- as.integer(id_order[focal_pos])
  nbr_ids   <- as.integer(id_order[nbr_pos])

  edge_dt <- data.table(focal_id = focal_ids, nbr_id = nbr_ids)
  rm(focal_pos, nbr_pos, focal_ids, nbr_ids)

  # Build a row-index lookup: (id, year) -> .row_idx
  row_map <- cell_data[, .(id, year, .row_idx)]
  setkey(row_map, id, year)

  # For each edge, we need to replicate across all years where BOTH

  # focal and neighbor exist. Use a merge-based approach:
  # Join edge_dt to row_map for focal side, then join for neighbor side.

  # Focal join: get all (focal_id, year, focal_row) combinations
  setkey(edge_dt, focal_id)
  # Expand: for each edge, find all years the focal cell has data
  focal_expanded <- row_map[edge_dt,
                            on = .(id = focal_id),
                            allow.cartesian = TRUE,
                            nomatch = NULL]
  # Columns: id (=focal_id), year, .row_idx (=focal_row), nbr_id
  setnames(focal_expanded, c("id", ".row_idx"), c("focal_id", "focal_row"))

  # Neighbor join: match (nbr_id, year) to get nbr_row
  setkey(focal_expanded, nbr_id, year)
  setkey(row_map, id, year)

  pair_dt <- row_map[focal_expanded,
                     on = .(id = nbr_id, year = year),
                     nomatch = NULL]
  # Columns: id (=nbr_id), year, .row_idx (=nbr_row), focal_id, focal_row
  setnames(pair_dt, c("id", ".row_idx"), c("nbr_id", "nbr_row"))

  pair_dt[, .(focal_row = as.integer(focal_row),
              nbr_row   = as.integer(nbr_row))]
}

cat("Building neighbor pair table...\n")
system.time({
  pair_dt <- build_neighbor_pairs_dt(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~38M rows, ~300 MB, built in 1-3 minutes
cat(sprintf("Neighbor pairs: %s rows\n", format(nrow(pair_dt), big.mark = ",")))


# ---- STEP 2: Compute all neighbor features (vectorized) --------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))

    # Extract the variable values for all neighbor rows
    pair_dt[, nbr_val := cell_data[[var_name]][nbr_row]]

    # Aggregate: max, min, mean per focal_row (excluding NAs)
    stats <- pair_dt[!is.na(nbr_val),
                     .(v_max  = max(nbr_val),
                       v_min  = min(nbr_val),
                       v_mean = mean(nbr_val)),
                     by = focal_row]

    # Assign back to cell_data using := (in-place, no copy)
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")

    # Initialize with NA
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Fill in computed values
    cell_data[stats$focal_row, (max_col)  := stats$v_max]
    cell_data[stats$focal_row, (min_col)  := stats$v_min]
    cell_data[stats$focal_row, (mean_col) := stats$v_mean]

    # Clean up the temporary column
    pair_dt[, nbr_val := NULL]
    rm(stats)
  }
})
# Expected: ~2-5 minutes total for all 5 variables

# Clean up
cell_data[, .row_idx := NULL]
rm(pair_dt)
gc()


# ---- STEP 3: Random Forest Prediction (chunked, memory-safe) ---------------

# Detect model type and set up prediction accordingly
predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)

  # Determine model class
  model_class <- class(model)[1]
  cat(sprintf("Model class: %s | Rows: %s | Chunks: %d\n",
              model_class, format(n, big.mark = ","), n_chunks))

  # For ranger: ensure num.threads is set for parallel prediction
  is_ranger <- inherits(model, "ranger")

  # Pre-convert to matrix if all predictors are numeric (avoids per-chunk coercion)
  # Identify predictor columns (exclude response, id, year if present)
  # This depends on your model; adjust as needed.

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    idx       <- start_row:end_row

    chunk <- newdata[idx, ]

    if (is_ranger) {
      pred <- predict(model, data = chunk, num.threads = parallel::detectCores())
      predictions[idx] <- pred$predictions
    } else {
      # randomForest
      pred <- predict(model, newdata = chunk)
      predictions[idx] <- as.numeric(pred)
    }

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %s-%s)\n",
                  i, n_chunks,
                  format(start_row, big.mark = ","),
                  format(end_row, big.mark = ",")))
    }
  }

  predictions
}

# Prepare prediction data: select only predictor columns, convert to data.table
# Adjust 'predictor_cols' to match your trained model's expected features
# Example: if the model was trained on all columns except "id", "year", "gdp":
response_col <- "gdp"  # adjust to your actual response variable name
exclude_cols <- c("id", "year", response_col)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Verify all expected predictors are present
if (inherits(rf_model, "ranger")) {
  expected_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  expected_vars <- rownames(rf_model$importance)
} else {
  expected_vars <- predictor_cols
}

missing_vars <- setdiff(expected_vars, names(cell_data))
if (length(missing_vars) > 0) {
  warning(sprintf("Missing predictor variables: %s",
                  paste(missing_vars, collapse = ", ")))
}

# Build prediction input (only needed columns, as data.table)
pred_input <- cell_data[, .SD, .SDcols = intersect(expected_vars, names(cell_data))]

cat("Starting Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model      = rf_model,
    newdata    = pred_input,
    chunk_size = 500000L
  )]
})

rm(pred_input)
gc()

cat("Pipeline complete.\n")
cat(sprintf("Predictions: %s rows\n",
            format(sum(!is.na(cell_data$predicted_gdp)), big.mark = ",")))


# =============================================================================
# OPTIONAL: If model is randomForest and prediction is still too slow,
# convert prediction input to a matrix (avoids internal data.frame -> matrix
# coercion on every predict() call)
# =============================================================================

# pred_matrix <- as.matrix(pred_input)
# Then in predict_rf_chunked, pass pred_matrix[idx, , drop = FALSE] to predict()
```

---

## 4. Summary of Key Changes

| Original | Optimized | Why |
|---|---|---|
| `paste(id, year)` string keys for 6.46M rows | Integer-keyed `data.table` joins on `(id, year)` | Eliminates millions of string allocations and hash lookups |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Vectorized `data.table` cartesian join + merge | Single C-level operation replaces 6.46M R function calls |
| `lapply` over 6.46M rows in `compute_neighbor_stats` (×5) | `data.table` grouped aggregation `[, .(max, min, mean), by = focal_row]` | Vectorized C-level groupby replaces 32.3M R function calls |
| `do.call(rbind, list_of_6.46M_vectors)` | `data.table` returns result directly as columns | No row-by-row matrix assembly |
| `data.frame$new_col <- ...` (copy-on-modify, ~5.7 GB copy each time) | `data.table[, (col) := val]` in-place assignment | Zero-copy column addition; stays within 16 GB RAM |
| Possible single `predict()` call on 6.46M rows | Chunked prediction (500K rows/chunk) with parallel threads | Controls peak memory; enables `ranger` multi-threading |
| Possible `data.frame` input to `predict()` | Pre-selected `data.table` / optional matrix conversion | Avoids repeated internal coercion |

**Estimated total runtime: ~15–40 minutes** (down from 86+ hours), depending on RF model size and whether `ranger` or `randomForest` is used. The trained model and original GDP estimand are fully preserved — no retraining or transformation of the response variable occurs.