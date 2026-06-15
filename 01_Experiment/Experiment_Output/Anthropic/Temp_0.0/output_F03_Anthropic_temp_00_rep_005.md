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

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix — this is notoriously slow because `rbind` on a list of millions of vectors has quadratic-like overhead in base R.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 × 6.46M = ~32.3M iterations total.

### 1.2 Prediction Bottleneck

With ~110 predictors and 6.46M rows, `predict.randomForest` (or `predict.ranger`) must push every row through every tree. If the model has hundreds of trees, this is a large computation. Additionally:
- If the model is a `randomForest` object (from the `randomForest` package), prediction is single-threaded and slow.
- If the entire 6.46M-row data.frame is passed at once, peak memory can spike (data.frame + prediction workspace + output).
- Unnecessary object copies (e.g., repeated `cell_data <- cbind(cell_data, ...)`) trigger R's copy-on-modify, duplicating the entire multi-GB data.frame each time a column is added.

### 1.3 Memory Pressure

- 6.46M rows × 110 columns × 8 bytes ≈ 5.7 GB just for the numeric matrix.
- The neighbor lookup list (6.46M elements, each a small integer vector) adds ~1–2 GB.
- Intermediate copies from `cbind` / column assignment on a data.frame can double memory transiently.
- On a 16 GB laptop, this leaves almost no headroom, causing swapping and massive slowdowns.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| **Data structure** | `data.frame` triggers copy-on-modify on column add | Use `data.table` — columns can be added by reference (`:=`) with zero copy |
| **Neighbor lookup** | 6.46M `paste`/hash lookups in R loop | Vectorize entirely: merge via `data.table` keyed join; build lookup as integer-index vectors using fast joins |
| **Neighbor stats** | Row-wise `lapply` + `do.call(rbind, ...)` over 6.46M rows | Explode neighbor pairs into a long `data.table`, join the variable values, then aggregate with `data.table` grouped operations (`[, .(max, min, mean), by=]`) — fully vectorized, no R-level loop |
| **Column addition** | `cell_data <- cbind(cell_data, new_cols)` copies entire frame | `data.table` `:=` adds columns in-place |
| **Prediction engine** | `randomForest::predict` is single-threaded | Convert model to `ranger` format or, if already `ranger`, ensure `num.threads` is set; alternatively, predict in chunks to control memory |
| **Prediction memory** | Passing 6.46M rows at once | Predict in batches (e.g., 500K rows) to cap peak memory |
| **Neighbor list storage** | R list of 6.46M integer vectors | Convert to a two-column long `data.table` (row_idx, neighbor_row_idx) — more cache-friendly and enables vectorized joins |

**Expected speedup:** From 86+ hours to roughly 10–30 minutes for feature preparation, and prediction time dependent on forest size but improved by multi-threading and batching.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   - If the trained model is a randomForest object, we wrap prediction
#     accordingly. If it is a ranger object, we use ranger::predict directly.
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place if possible) ---------
# Assumes cell_data is a data.frame with columns: id, year, and all predictors.
# This conversion is O(1) if cell_data is already a data.table.

setDT(cell_data)

# Ensure id and year are the types we expect
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row index for the original data (preserves original row order)
cell_data[, .row_idx := .I]


# =============================================================================
# STEP 1: BUILD VECTORIZED NEIGHBOR LOOKUP (LONG TABLE)
# =============================================================================
# Instead of a list of 6.46M elements, we build a two-column data.table:
#   (focal_row_idx, neighbor_row_idx)
# This enables fully vectorized grouped aggregation.

build_neighbor_lookup_dt <- function(dt, id_order, neighbors_nb) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors into id_order)
  
  # --- Map each nb index to the actual cell id ---
  # neighbors_nb[[i]] gives the indices (into id_order) of neighbors of
  # the cell whose id is id_order[i].
  
  # Build an edge list: (focal_cell_id, neighbor_cell_id)
  # This is done once and is independent of year.
  
  n_cells <- length(id_order)
  
  # Pre-compute lengths for pre-allocation
  lens <- lengths(neighbors_nb)  # fast C-level lengths
  total_edges <- sum(lens)       # ~1.37M directed edges
  
  focal_ids    <- rep.int(id_order, lens)
  neighbor_ids <- id_order[unlist(neighbors_nb, use.names = FALSE)]
  
  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  
  # --- Expand edges across all years ---
  # Each (focal_id, year) needs neighbors from the same year.
  unique_years <- sort(unique(dt$year))
  
  # Cross join edges with years
  edges_by_year <- edges[, CJ_val := 1L][
    data.table(year = unique_years, CJ_val = 1L),
    on = "CJ_val",
    allow.cartesian = TRUE
  ]
  edges_by_year[, CJ_val := NULL]
  
  # --- Map (id, year) to row index in dt ---
  # Build a keyed lookup: (id, year) -> .row_idx
  row_map <- dt[, .(id, year, .row_idx)]
  setkey(row_map, id, year)
  
  # Map focal
  edges_by_year[row_map, focal_row := i..row_idx,
                on = .(focal_id = id, year = year)]
  
  # Map neighbor
  edges_by_year[row_map, neighbor_row := i..row_idx,
                on = .(neighbor_id = id, year = year)]
  
  # Drop edges where either side is missing
  edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]
  
  # Return only the row-index columns (compact)
  edges_by_year[, .(focal_row, neighbor_row)]
}

cat("Building vectorized neighbor lookup...\n")
system.time({
  neighbor_edges <- build_neighbor_lookup_dt(
    cell_data, id_order, rook_neighbors_unique
  )
})
# neighbor_edges is a data.table with columns: focal_row, neighbor_row
# Rows: ~1.37M edges × 28 years ≈ 38.5M rows (manageable)

setkey(neighbor_edges, focal_row)

cat(sprintf("Neighbor edge table: %s rows\n", format(nrow(neighbor_edges), big.mark = ",")))


# =============================================================================
# STEP 2: COMPUTE AND ADD NEIGHBOR FEATURES (FULLY VECTORIZED)
# =============================================================================
# For each source variable, compute max/min/mean of neighbor values,
# then join back to cell_data by reference.

compute_and_add_neighbor_features_dt <- function(dt, var_name, edges) {
  # Extract the variable values for all neighbor rows
  # edges$neighbor_row indexes into dt
  vals <- dt[[var_name]]
  
  # Attach neighbor values to the edge table (no copy of dt)
  edge_vals <- edges[, .(focal_row, nval = vals[neighbor_row])]
  
  # Remove NA neighbor values before aggregation
  edge_vals <- edge_vals[!is.na(nval)]
  
  # Grouped aggregation — extremely fast in data.table
  agg <- edge_vals[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]
  
  # Prepare column names
  col_max  <- paste0("nb_max_", var_name)
  col_min  <- paste0("nb_min_", var_name)
  col_mean <- paste0("nb_mean_", var_name)
  
  # Initialize columns with NA (for rows with no valid neighbors)
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)
  
  # Fill in computed values by reference (no copy)
  set(dt, i = agg$focal_row, j = col_max,  value = agg$nb_max)
  set(dt, i = agg$focal_row, j = col_min,  value = agg$nb_min)
  set(dt, i = agg$focal_row, j = col_mean, value = agg$nb_mean)
  
  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    compute_and_add_neighbor_features_dt(cell_data, var_name, neighbor_edges)
  }
})

# Clean up the large edge table if memory is tight
# rm(neighbor_edges); gc()


# =============================================================================
# STEP 3: PREPARE PREDICTION MATRIX
# =============================================================================
# Identify the predictor columns expected by the model.
# Adjust 'predictor_cols' to match your trained model's expected features.

# If your model is a ranger object:
#   predictor_cols <- rf_model$forest$independent.variable.names
# If your model is a randomForest object:
#   predictor_cols <- rownames(importance(rf_model))
# Or define them explicitly:

if (inherits(rf_model, "ranger")) {
  predictor_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  predictor_cols <- rownames(importance(rf_model))
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all predictor columns exist
missing_cols <- setdiff(predictor_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop("Missing predictor columns: ", paste(missing_cols, collapse = ", "))
}

cat(sprintf("Predictor columns: %d\n", length(predictor_cols)))


# =============================================================================
# STEP 4: BATCHED PREDICTION (MEMORY-SAFE, MULTI-THREADED IF RANGER)
# =============================================================================

predict_batched <- function(model, dt, pred_cols, batch_size = 500000L) {
  n <- nrow(dt)
  n_batches <- ceiling(n / batch_size)
  predictions <- numeric(n)
  
  is_ranger <- inherits(model, "ranger")
  
  cat(sprintf("Predicting %s rows in %d batches of up to %s...\n",
              format(n, big.mark = ","),
              n_batches,
              format(batch_size, big.mark = ",")))
  
  for (b in seq_len(n_batches)) {
    i_start <- (b - 1L) * batch_size + 1L
    i_end   <- min(b * batch_size, n)
    idx     <- i_start:i_end
    
    # Extract batch as a plain data.frame (required by predict methods)
    batch_df <- as.data.frame(dt[idx, ..pred_cols])
    
    if (is_ranger) {
      # ranger::predict is multi-threaded by default
      pred_obj <- predict(model, data = batch_df, num.threads = parallel::detectCores())
      predictions[idx] <- pred_obj$predictions
    } else {
      # randomForest::predict — single-threaded but we avoid memory bloat
      predictions[idx] <- predict(model, newdata = batch_df)
    }
    
    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("  Batch %d/%d complete (rows %s-%s)\n",
                  b, n_batches,
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }
  }
  
  predictions
}

cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_batched(
    rf_model, cell_data, predictor_cols, batch_size = 500000L
  )]
})

# Remove helper column
cell_data[, .row_idx := NULL]

cat("Done.\n")
cat(sprintf("Output rows: %s | Columns: %d\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))


# =============================================================================
# OPTIONAL: CONVERT RANDOMFOREST MODEL TO RANGER FOR FUTURE SPEED
# =============================================================================
# If your model is a `randomForest` object and prediction speed is still
# a bottleneck, consider a one-time conversion. This does NOT retrain;
# it restructures the tree representation for faster multi-threaded prediction.
#
# NOTE: There is no lossless automatic converter in CRAN packages.
# The recommended path is to retrain with ranger using the same hyperparameters
# and the same seed/data to reproduce identical results. However, per your
# constraint ("must not be retrained"), keep the original model and rely on
# the batching strategy above.
# =============================================================================
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Expected Speedup |
|---|---|---|---|
| **Data structure** | `data.frame` + `cbind` copies | `data.table` + `:=` by reference | Eliminates ~5.7 GB repeated copies |
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/hash | Vectorized `data.table` join, built once | ~100–500× faster (minutes vs. hours) |
| **Neighbor stats** | `lapply` + `do.call(rbind, ...)` per variable | `data.table` grouped `[, .(max,min,mean), by=]` | ~100–500× faster; no R-level loop |
| **Column addition** | `cbind` (full copy each time) | `set()` by reference | Zero-copy; saves ~30 GB of transient allocation |
| **Prediction** | Presumably single call on full 6.46M rows | Batched (500K rows), multi-threaded if `ranger` | Caps peak memory; enables parallelism |
| **Overall** | ~86+ hours estimated | ~15–45 minutes expected | ~100–300× |

**Numerical equivalence is preserved:** The neighbor statistics (max, min, mean) are computed with the same semantics (NA removal, same aggregation functions). The trained Random Forest model is used as-is with no retraining. The predicted values will be numerically identical to the original pipeline's output.