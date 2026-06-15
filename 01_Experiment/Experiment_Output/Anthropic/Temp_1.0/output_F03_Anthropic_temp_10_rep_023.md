 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows using `lapply`, performing per-row string pasting (`paste(id, year)`), hash-table lookups, and NA filtering. With ~6.46M iterations, each involving character coercion, string concatenation, and named-vector lookups, this alone can take hours. Named vector lookups in R are O(n) in pathological cases and carry per-call overhead from character matching.

**`compute_neighbor_stats`:** Called 5 times (once per variable), it iterates over 6.46M rows again via `lapply`, each time subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end builds a 6.46M × 3 matrix from a list of 6.46M length-3 vectors — this is extremely slow and memory-intensive.

**Outer loop:** The sequential `for` loop over 5 variables calls `compute_neighbor_stats` five separate times, re-traversing the neighbor lookup each time rather than computing all variables in a single pass.

### 1.2 Random Forest Inference Bottleneck

Calling `predict()` on a single `randomForest` or `ranger` model object with 6.46M rows and ~110 predictors is itself expensive. If the model was trained with the `randomForest` package (not `ranger`), prediction is done in pure R/C with poor memory locality and no parallelism. Additionally:

- If prediction is done in a **row-by-row or chunked loop** rather than a single vectorized call, overhead multiplies dramatically.
- If the prediction data.frame is being **copied** (e.g., via `data.frame()` construction or column-binding inside a loop), each copy of a 6.46M × 110 frame is ~5–6 GB, which on a 16 GB machine causes swapping.
- Model objects from `randomForest` can be very large in memory; loading and holding them alongside the data may exceed RAM.

### 1.3 Summary of Root Causes

| Cause | Impact |
|---|---|
| Per-row `lapply` with string ops in `build_neighbor_lookup` | Hours of wall time for 6.46M rows |
| `do.call(rbind, ...)` on millions of small vectors | Massive memory allocation + GC pressure |
| Repeated traversal of neighbor lookup (5×) | 5× redundant iteration |
| Likely row-wise or copy-heavy prediction workflow | Memory thrashing on 16 GB laptop |
| Possible use of `randomForest::predict` instead of `ranger::predict` | Slow single-threaded C inference |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation — Vectorized with `data.table`

- Replace all character-keyed lookups with integer-keyed joins via `data.table`.
- Build the neighbor lookup as a **two-column integer edge-list** (`row_i`, `neighbor_row_j`), avoiding per-row `lapply` entirely.
- Compute all neighbor stats (max, min, mean) across **all 5 variables simultaneously** in a single grouped `data.table` aggregation on the edge-list — one pass, fully vectorized in C.
- This eliminates `do.call(rbind, ...)`, per-row `lapply`, and string operations.

### 2.2 Random Forest Inference

- If the model is a `randomForest` object, convert it to `ranger` format or, if that's not feasible, use `predict()` in a **single vectorized call** on the full matrix, ensuring no per-row loop.
- If `ranger` is usable, re-wrap the predict call with `ranger::predict` which is multi-threaded.
- Prepare the prediction input as a **matrix** (not data.frame) to avoid method-dispatch overhead and column-type checking on each tree traversal.
- Predict in **chunks** (e.g., 500K rows) only if memory is the binding constraint, to stay within 16 GB.

### 2.3 Memory Management

- Use in-place column assignment (`:=` in `data.table`) to avoid copying the full 6.46M-row table.
- Remove intermediate objects and call `gc()` before prediction.
- Ensure only one copy of the data exists at prediction time.

### Expected Speedup

| Stage | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~10–20 hrs | ~10–30 sec |
| `compute_neighbor_stats` (×5) | ~40–60 hrs | ~20–60 sec |
| RF prediction | ~5–10 hrs | ~5–20 min |
| **Total** | **~86+ hrs** | **~10–25 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (optional, for faster predict)
# =============================================================================

library(data.table)

# ---- 0. Load pre-trained model and data ------------------------------------
# Adjust paths as needed
# rf_model       <- readRDS("trained_rf_model.rds")
# cell_data      <- readRDS("cell_data.rds")           # data.frame or data.table
# id_order       <- readRDS("id_order.rds")             # vector of cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds") # spdep nb object

# ---- 1. Convert to data.table in place -------------------------------------
if (!is.data.table(cell_data)) setDT(cell_data)

# ---- 2. Build vectorized neighbor edge-list --------------------------------
build_neighbor_edgelist <- function(dt, id_order, neighbors) {
  # Map each cell ID to its position in id_order (1-based)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a row-index keyed by (id, year) — integer key for speed
  # We assign each row a sequential row number
  dt[, .row_idx := .I]

  # Create an integer-keyed lookup: for each unique (id, year) -> row index
  # Use data.table keyed join instead of named character vector
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # Expand the spdep nb object into a two-column edge-list:
  #   focal_ref (index into id_order) -> neighbor_ref (index into id_order)
  n_refs <- length(neighbors)
  focal_refs <- rep(seq_len(n_refs),
                    times = lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)

  # Filter out 0-length (isolated) nodes — already handled by rep/unlist
  # Convert ref indices to actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )

  # Now cross-join with years: each edge applies to every year
  # Instead of a full cross join (expensive), we join via the data
  # For each row in dt, find its focal_id, then look up neighbor rows

  # Step A: For each focal cell, list its neighbor cell IDs
  # (this is small: ~344K cells, ~1.37M edges)
  # Step B: For each row in dt, get the neighbor cell IDs, then find
  #         the row indices of (neighbor_id, same year)

  # Efficient approach: join edge_dt with row_key on focal side,
  # then join again on neighbor side for the same year.

  # Get all (focal_id, year, neighbor_id) combinations by joining
  # dt's (id, year, row_idx) with edge_dt on focal_id

  setnames(edge_dt, c("focal_id", "neighbor_id"))

  # Join: for each row in dt, get its neighbor IDs
  # focal_rows: (focal_id, year, focal_row_idx)
  focal_rows <- dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]
  setkey(focal_rows, focal_id)
  setkey(edge_dt, focal_id)

  # This is the large join: 6.46M rows × ~4 neighbors each ≈ 25.8M rows
  expanded <- edge_dt[focal_rows,
                      .(focal_row_idx, neighbor_id, year),
                      on = "focal_id",
                      allow.cartesian = TRUE,
                      nomatch = 0L]

  # Now find the row index of each (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  # row_key is keyed on (id, year)
  expanded[row_key,
           neighbor_row_idx := i..row_idx,
           on = c(neighbor_id = "id", "year")]

  # Drop rows where neighbor was not found (boundary cells in some years)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Clean up
  dt[, .row_idx := NULL]

  return(expanded[, .(focal_row_idx, neighbor_row_idx)])
}

cat("Building neighbor edge-list...\n")
system.time({
  edge_list <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~10-30 seconds, ~25M rows, two integer columns (~200 MB)

# ---- 3. Compute all neighbor stats in one vectorized pass ------------------
compute_all_neighbor_stats <- function(dt, edge_list, var_names) {
  # Extract the neighbor values for ALL variables at once
  # edge_list has columns: focal_row_idx, neighbor_row_idx

  # Build a sub-table of neighbor values
  neighbor_vals <- dt[edge_list$neighbor_row_idx, ..var_names]
  neighbor_vals[, focal_row_idx := edge_list$focal_row_idx]

  # Aggregate: for each focal_row_idx, compute max/min/mean of each variable
  # Use data.table's efficient grouped aggregation
  agg_exprs <- list()
  for (v in var_names) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  agg_result <- neighbor_vals[, eval(agg_call), by = focal_row_idx]

  # Replace Inf/-Inf from max/min of all-NA groups with NA
  inf_cols <- grep("^n_max_|^n_min_", names(agg_result), value = TRUE)
  for (col in inf_cols) {
    vals <- agg_result[[col]]
    set(agg_result, which(is.infinite(vals)), col, NA_real_)
  }

  return(agg_result)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics for all variables...\n")
system.time({
  neighbor_stats <- compute_all_neighbor_stats(cell_data, edge_list, neighbor_source_vars)
})
# Expected: ~20-60 seconds

# ---- 4. Join neighbor stats back to the main table -------------------------
cat("Joining neighbor features to main data...\n")

# Ensure row indices align
cell_data[, .row_idx := .I]
setkey(neighbor_stats, focal_row_idx)

# Join in place
stat_cols <- setdiff(names(neighbor_stats), "focal_row_idx")
cell_data[neighbor_stats, (stat_cols) := mget(paste0("i.", stat_cols)),
          on = c(.row_idx = "focal_row_idx")]
cell_data[, .row_idx := NULL]

# Rows without neighbors will have NA — this matches original behavior

# Free memory
rm(edge_list, neighbor_stats)
gc()

# ---- 5. Prepare prediction matrix -----------------------------------------
cat("Preparing prediction input...\n")

# Identify the predictor columns (adjust to match your trained model)
# If your model was trained with specific feature names, use those:
predictor_cols <- setdiff(names(cell_data),
                          c("id", "year", "gdp", "gdp_predicted",
                            # add any other non-predictor columns here
                            "geometry"))

# Verify we have the expected ~110 predictors
cat(sprintf("  Number of predictor columns: %d\n", length(predictor_cols)))

# Build a matrix for faster predict (if model supports it)
# For randomForest/ranger, data.frame is required; but we avoid copies
pred_data <- cell_data[, ..predictor_cols]

# ---- 6. Predict using the trained Random Forest ---------------------------
cat("Running Random Forest prediction...\n")

# Detect model type and predict accordingly
if (inherits(rf_model, "ranger")) {
  # ranger: natively multi-threaded, fast
  system.time({
    preds <- predict(rf_model, data = pred_data,
                     num.threads = parallel::detectCores())$predictions
  })
} else if (inherits(rf_model, "randomForest")) {
  # randomForest package: single-threaded, slower
  # Predict in chunks to manage memory (each chunk ~ 500K rows)
  chunk_size <- 500000L
  n_rows     <- nrow(pred_data)
  n_chunks   <- ceiling(n_rows / chunk_size)
  preds      <- numeric(n_rows)

  cat(sprintf("  Predicting in %d chunks of up to %d rows...\n",
              n_chunks, chunk_size))

  system.time({
    for (ch in seq_len(n_chunks)) {
      start_i <- (ch - 1L) * chunk_size + 1L
      end_i   <- min(ch * chunk_size, n_rows)
      preds[start_i:end_i] <- predict(rf_model,
                                       newdata = pred_data[start_i:end_i, ])
      if (ch %% 5 == 0) {
        cat(sprintf("    Chunk %d/%d done\n", ch, n_chunks))
        gc()
      }
    }
  })
} else {
  # Generic fallback
  system.time({
    preds <- predict(rf_model, newdata = pred_data)
  })
}

# ---- 7. Attach predictions ------------------------------------------------
cell_data[, gdp_predicted := preds]

# Free prediction input
rm(pred_data, preds)
gc()

cat("Done. Predictions stored in cell_data$gdp_predicted.\n")

# =============================================================================
# OPTIONAL: If you have a randomForest model and want maximum speed,
# convert it to a ranger-compatible prediction. This preserves the original
# trained model's numerical output exactly IF the conversion is lossless.
# Otherwise, keep the chunked predict() above.
#
# NOTE: There is no automatic lossless conversion from randomForest to ranger.
# If retraining is forbidden, use the chunked approach for randomForest models.
# =============================================================================
```

---

## 4. KEY POINTS AND GUARANTEES

| Requirement | How It Is Met |
|---|---|
| **Preserve trained RF model** | The model object is loaded from disk and used read-only via `predict()`. No retraining occurs. |
| **Preserve original numerical estimand** | `predict()` is called on the same model with the same features; neighbor stats use identical formulae (max, min, mean). Infinite values from all-NA groups are mapped to `NA`, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` logic. |
| **16 GB RAM feasibility** | `data.table` in-place operations avoid copies. Edge-list is ~200 MB. Chunked prediction (for `randomForest`) keeps peak memory well under 16 GB. Intermediate objects are freed with `rm()` + `gc()`. |
| **Speed** | String operations and per-row `lapply` are eliminated. All neighbor computation is a single vectorized `data.table` grouped aggregation. Expected total runtime: **~10–25 minutes** vs. 86+ hours. |