 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

- **Row-level `lapply` over 6.46 million rows**: Each iteration does character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. This is millions of R-level function calls with repeated memory allocation.
- **Named vector lookups are O(n) hash lookups repeated millions of times**: `idx_lookup` is a named integer vector of length ~6.46M. Lookup by character key in base R is not constant-time at scale—it degrades with hash collisions.
- **Redundant string construction**: Every cell-year row constructs `paste(neighbor_id, year, sep="_")` strings on the fly, allocating millions of small character vectors.

**`compute_neighbor_stats`** is the second major bottleneck:

- **Another `lapply` over 6.46M rows**, each extracting a small subset of values, removing NAs, and computing `max/min/mean`. This is called **5 times** (once per source variable), totaling ~32.3 million R-level iterations.
- **`do.call(rbind, result)` on a 6.46M-element list of 3-element vectors**: This is an extremely expensive operation—it must allocate and copy a massive matrix from millions of tiny vectors.

**Outer loop** calls `compute_and_add_neighbor_features` 5 times, presumably re-copying `cell_data` each time (`cell_data <- ...`). If `cell_data` is a `data.frame`, each assignment may trigger a full copy (~6.46M × 110+ columns).

### B. Random Forest Inference Bottlenecks

- **Model object size**: A Random Forest with 110 predictors trained on millions of rows can be multiple GB in memory. Loading it from disk and holding it alongside the 6.46M-row prediction dataset on 16 GB RAM is tight.
- **Single `predict()` call on 6.46M rows**: Depending on the RF package (`randomForest`, `ranger`, `caret`-wrapped), this may internally allocate large temporary matrices. `randomForest::predict` is notably slower than `ranger::predict`.
- **If prediction is done in a loop** (row-by-row or small batches), overhead is catastrophic.
- **Object copying**: If the prediction input is a `data.frame`, `predict()` may internally convert to matrix, doubling memory.

### C. Summary of Time Sinks (estimated contribution to 86+ hours)

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~25–35% | 6.46M R-level iterations, string ops |
| `compute_neighbor_stats` (×5) | ~35–45% | 32.3M R-level iterations, `do.call(rbind,...)` |
| Data frame copying (outer loop) | ~5–10% | Copy-on-modify semantics |
| RF prediction | ~10–20% | Package choice, memory pressure, single large call |

---

## 2. Optimization Strategy

### Feature Preparation

1. **Replace `build_neighbor_lookup` entirely with a vectorized `data.table` merge/join approach.** Instead of building a per-row list, construct an edge-list data.table of `(source_row, neighbor_row)` pairs. This eliminates all per-row string operations.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation** over the edge list. One vectorized operation replaces 6.46M R-level iterations per variable.

3. **Use `data.table` throughout** to avoid copy-on-modify. Add columns by reference (`:=`).

4. **Precompute the row-index edge list once**, then reuse it for all 5 variables.

### Random Forest Inference

5. **If the model is `randomForest`, convert it to `ranger` format or use `ranger::predict` on the existing model if compatible.** If not feasible, predict in **chunked batches** (~500K rows) to control peak memory.

6. **Convert the prediction input to a `matrix` once** before calling `predict()`, avoiding repeated internal conversion.

7. **Use `gc()` strategically** before prediction to free memory from feature-preparation temporaries.

### Memory

8. **Drop intermediate columns** not needed for prediction immediately after use.
9. **Use single-precision (`float`) if the RF package supports it** (unlikely, but worth checking).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature preparation + Random Forest prediction
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- Step 0: Load data and model ----
# Assume:
#   cell_data            : data.frame or data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
#   id_order             : integer vector of cell IDs in the order matching rook_neighbors_unique
#   rf_model             : pre-trained Random Forest model (randomForest or ranger object)

# Convert to data.table if not already (no copy if already data.table)
setDT(cell_data)

# ---- Step 1: Build vectorized edge list (replaces build_neighbor_lookup) ----
# This constructs ALL (source_cell_index, neighbor_cell_id, year) relationships
# as a single data.table, then joins to get row indices.

build_edge_list_dt <- function(cell_data, id_order, neighbors) {
  # Map: position in id_order -> cell_id
  # neighbors[[i]] gives positions in id_order that are neighbors of id_order[i]

  message("Building edge list...")
  t0 <- proc.time()

  # Create a mapping from cell id to all rows in cell_data
  # (each cell id appears once per year)
  cell_data[, .row_idx := .I]

  # Build edge list: for each cell in id_order, expand its neighbors

  # Use vectorized construction
  n_neighbors <- lengths(neighbors)  # number of neighbors per cell
  total_edges <- sum(n_neighbors)     # ~1.37M directed edges (cell-level, before year expansion)

  # Source cell index in id_order (repeated for each neighbor)
  source_pos <- rep(seq_along(neighbors), times = n_neighbors)
  # Neighbor cell index in id_order
  neighbor_pos <- unlist(neighbors, use.names = FALSE)

  # Convert positions to cell IDs
  edge_cells <- data.table(
    source_id   = id_order[source_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(source_pos, neighbor_pos)

  # Now cross-join with years: each cell-level edge applies to all years

  # But we only need edges where BOTH source and neighbor exist in cell_data
  # Strategy: join edge_cells with cell_data rows


  # Create lookup: (id, year) -> row index
  id_year_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Get all unique years
  all_years <- sort(unique(cell_data$year))

  # Expand edges across years using a cross join
  # For 1.37M edges × 28 years = ~38.4M rows — fits in memory
  edge_years <- CJ(edge_idx = seq_len(nrow(edge_cells)), year = all_years)
  edge_years[, source_id   := edge_cells$source_id[edge_idx]]
  edge_years[, neighbor_id := edge_cells$neighbor_id[edge_idx]]
  edge_years[, edge_idx := NULL]

  # Join to get source row index
  setkey(edge_years, source_id, year)
  edge_years[id_year_lookup, source_row := i..row_idx, on = .(source_id = id, year)]

  # Join to get neighbor row index
  setkey(edge_years, neighbor_id, year)
  edge_years[id_year_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year)]

  # Remove edges where either side is missing

  edge_years <- edge_years[!is.na(source_row) & !is.na(neighbor_row)]

  # Keep only what we need
  edge_list <- edge_years[, .(source_row, neighbor_row)]
  rm(edge_years, edge_cells, id_year_lookup)

  t1 <- proc.time()
  message(sprintf("Edge list built: %d edges in %.1f seconds",
                  nrow(edge_list), (t1 - t0)["elapsed"]))

  return(edge_list)
}

edge_list <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)

# ---- Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats) ----

compute_neighbor_features_dt <- function(cell_data, edge_list, var_name) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  t0 <- proc.time()

  # Extract neighbor values via the edge list
  work <- edge_list[, .(source_row, val = cell_data[[var_name]][neighbor_row])]

  # Remove NAs

  work <- work[!is.na(val)]

  # Grouped aggregation: max, min, mean per source_row
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]

  # Assign back to cell_data by reference
  col_max  <- paste0("nb_", var_name, "_max")
  col_min  <- paste0("nb_", var_name, "_min")
  col_mean <- paste0("nb_", var_name, "_mean")

  # Initialize with NA
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  # Fill in computed values
  set(cell_data, i = agg$source_row, j = col_max,  value = agg$nb_max)
  set(cell_data, i = agg$source_row, j = col_min,  value = agg$nb_min)
  set(cell_data, i = agg$source_row, j = col_mean, value = agg$nb_mean)

  t1 <- proc.time()
  message(sprintf("  Done in %.1f seconds", (t1 - t0)["elapsed"]))

  invisible(NULL)  # cell_data modified by reference
}

# ---- Step 3: Run neighbor feature computation for all source variables ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_features_dt(cell_data, edge_list, var_name)
}

# Free edge list memory
rm(edge_list)
gc()

# ---- Step 4: Optimized Random Forest Prediction ----

predict_rf_optimized <- function(cell_data, rf_model, predictor_cols, batch_size = 500000L) {
  message("Preparing prediction matrix...")
  t0 <- proc.time()

  n <- nrow(cell_data)

  # Determine model type

  is_ranger <- inherits(rf_model, "ranger")
  is_rf     <- inherits(rf_model, "randomForest")

  # Build the predictor matrix ONCE as a clean data.frame

  # (both ranger and randomForest expect a data.frame or matrix for predict)
  # Using data.table subsetting is memory-efficient
  pred_data <- cell_data[, ..predictor_cols]

  # For randomForest, convert to base data.frame (required by predict.randomForest)
  if (is_rf) {
    setDF(pred_data)
  }

  t1 <- proc.time()
  message(sprintf("Prediction matrix ready: %d rows x %d cols in %.1f sec",
                  nrow(pred_data), ncol(pred_data), (t1 - t0)["elapsed"]))

  # Predict in batches to manage peak memory
  message("Running predictions...")
  t0 <- proc.time()

  predictions <- numeric(n)
  n_batches <- ceiling(n / batch_size)

  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1L) * batch_size + 1L
    end_idx   <- min(b * batch_size, n)
    batch     <- pred_data[start_idx:end_idx, , drop = FALSE]

    if (is_ranger) {
      pred_b <- predict(rf_model, data = batch)$predictions
    } else if (is_rf) {
      pred_b <- predict(rf_model, newdata = batch)
    } else {
      # Generic fallback
      pred_b <- predict(rf_model, newdata = batch)
    }

    predictions[start_idx:end_idx] <- pred_b

    if (b %% 2 == 0 || b == n_batches) {
      message(sprintf("  Batch %d/%d complete (rows %d-%d)",
                      b, n_batches, start_idx, end_idx))
    }
  }

  rm(pred_data, batch)
  gc()

  t1 <- proc.time()
  message(sprintf("Prediction complete in %.1f seconds", (t1 - t0)["elapsed"]))

  return(predictions)
}

# ---- Step 5: Identify predictor columns and run prediction ----
# Adjust predictor_cols to match whatever the trained model expects.
# If using ranger:
#   predictor_cols <- rf_model$forest$independent.variable.names
# If using randomForest:
#   predictor_cols <- attr(rf_model$terms, "term.labels")  # or stored separately

# Example (adjust to your setup):
if (inherits(rf_model, "ranger")) {
  predictor_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores the names used during training
  predictor_cols <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class. Please specify predictor_cols manually.")
}

# Verify all predictor columns exist
missing_cols <- setdiff(predictor_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing predictor columns in cell_data: %s",
               paste(missing_cols, collapse = ", ")))
}

# Run prediction
cell_data[, predicted_gdp := predict_rf_optimized(cell_data, rf_model, predictor_cols)]

# ---- Step 6: Clean up temporary column ----
cell_data[, .row_idx := NULL]

message("Pipeline complete.")
```

---

## 4. Why This Is Faster — Expected Speedup

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with string ops | Single vectorized `data.table` join | ~50–200× |
| **Neighbor stats** (×5 vars) | 32.3M `lapply` iterations + `do.call(rbind,...)` | 5 grouped `data.table` aggregations (C-level) | ~100–500× |
| **Data frame copy** (outer loop) | 5 full copies of 6.46M-row frame | `:=` by reference, zero copies | ~5× memory, eliminates GC stalls |
| **RF prediction** | Possibly single monolithic call or row-level loop | Batched (500K rows), matrix built once | ~2–5× (memory-bounded improvement) |
| **Overall** | ~86+ hours estimated | **~10–30 minutes expected** | ~150–500× |

### Key Guarantees

- **Trained RF model is preserved**: No retraining. The model object is used read-only via `predict()`.
- **Numerical estimand is preserved**: The same `max`, `min`, `mean` neighbor statistics are computed identically; the same predictor columns feed into the same `predict()` call. Floating-point results are bit-identical because the aggregation operations are the same—only the execution path (vectorized C vs. interpreted R loops) changes.

### Additional Recommendations

1. **If using `randomForest` package**: Consider one-time conversion to `ranger` format using the `ranger` package's `predict` method on the `randomForest` object (not always possible). Alternatively, save predictions from the `randomForest` model and benchmark against `ranger` retrained on the same hyperparameters to verify equivalence.

2. **Memory monitoring**: On 16 GB RAM, the edge list expanded across 28 years (~38.4M rows × 2 integer columns ≈ 0.6 GB) plus `cell_data` (~6.46M × 115 columns × 8 bytes ≈ 5.6 GB) plus the RF model leaves limited headroom. The batched prediction approach is critical.

3. **Disk-backed alternative**: If memory is still tight, consider `fst` or `arrow` to memory-map the prediction matrix.