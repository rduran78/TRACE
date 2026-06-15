 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** — O(n) `lapply` over ~6.46 million rows, each performing:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`) — hash lookup per row.
- `paste()` to build neighbor keys — string allocation per neighbor per row.
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M) — this is **O(k)** per neighbor via partial-match hashing but with enormous constant overhead because R's named vector lookup on a 6.46M-element vector is essentially a repeated hash-table probe with string allocation.
- Net effect: ~6.46M iterations × ~4 neighbors avg = ~25.8M string constructions + hash lookups. Estimated wall time: **30–90 minutes**.

**`compute_neighbor_stats`** — For each of 5 variables:
- `lapply` over 6.46M rows, subsetting a numeric vector by integer index, removing NAs, computing max/min/mean.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors — this alone is an **O(n²)** memory-copy disaster because `rbind` on a list of vectors builds the matrix incrementally.
- Net effect per variable: **minutes to tens of minutes**, dominated by `do.call(rbind, ...)`.

**Outer loop** — Runs `compute_neighbor_stats` (and presumably column-binds results) 5 times → the `do.call(rbind, ...)` penalty is paid 5 times.

### B. Prediction Bottlenecks

With a trained Random Forest and 6.46M rows × 110 features:
- **Model loading**: `readRDS()` on a large `ranger`/`randomForest` object can take minutes and consume multiple GB.
- **Single `predict()` call on 6.46M rows**: If using `randomForest`, prediction is single-threaded and stores an internal copy of the data as a matrix — peak RAM ≈ 6.46M × 110 × 8 bytes × 2 copies ≈ **11.4 GB** just for the data, likely exceeding 16 GB with model overhead.
- **If looped row-by-row or in small batches**: R function-call overhead dominates; millions of iterations is fatal.
- **Object copying**: R's copy-on-modify semantics mean any `data$new_col <- ...` inside a loop triggers a full data.frame copy (~5.7 GB).

### C. Summary of Root Causes

| Rank | Bottleneck | Severity |
|------|-----------|----------|
| 1 | `do.call(rbind, list_of_vectors)` in `compute_neighbor_stats` | Critical — quasi-quadratic |
| 2 | String-key lookup in `build_neighbor_lookup` (6.46M × k) | High |
| 3 | `predict()` on full 6.46M rows at once (RAM) or row-by-row (overhead) | High |
| 4 | Repeated `data.frame` column assignment triggering copies | Moderate |
| 5 | Single-threaded prediction (`randomForest` package) | Moderate |

---

## 2. OPTIMIZATION STRATEGY

### Feature Preparation
1. **Replace `build_neighbor_lookup`** with a pure integer-index approach using `data.table` — build a `(cell_id, year) → row_index` hash table via `data.table` keyed join; vectorize neighbor expansion.
2. **Replace `compute_neighbor_stats`** — pre-allocate a matrix, use vectorized grouped operations via `data.table` instead of `lapply` + `do.call(rbind, ...)`.
3. **Avoid column-by-column data.frame mutation** — work entirely in `data.table` (in-place `:=` assignment, no copies).

### Prediction
4. **Chunk-based prediction** — split 6.46M rows into ~500K-row chunks to control peak RAM while avoiding per-row overhead.
5. **Use `ranger` for prediction if possible** — `ranger::predict` is multithreaded C++ and 5–20× faster than `randomForest::predict`. If the model is a `randomForest` object, we can still chunk it. If it's `ranger`, we enable `num.threads`.
6. **Load model once, predict in chunks, `gc()` between chunks**.

### Expected Speedup
- Feature preparation: from hours → **2–10 minutes**.
- Prediction: from hours → **10–45 minutes** (depending on model type and tree count).
- Total: from **86+ hours → under 1 hour**.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Dependencies
library(data.table)

# ---- Step 0: Convert to data.table (once, in-place) -------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist as expected
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Add a row index column (used for neighbor mapping)
cell_data[, .row_idx := .I]

# =============================================================================
# STEP 1: BUILD NEIGHBOR LOOKUP (vectorized, integer-only)
# =============================================================================
build_neighbor_lookup_fast <- function(dt, id_order, neighbors_nb) {
  # dt         : data.table with columns 'id', 'year', '.row_idx'
  # id_order   : integer vector of cell IDs in the order used by the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors)
  
  # --- 1a. Build directed edge list from nb object ---------------------------
  #     Each element neighbors_nb[[i]] is an integer vector of neighbor indices
  #     into id_order. We expand into a two-column edge table of cell IDs.
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb, use.names = FALSE)
  
  # Remove the 0-valued entries that spdep uses for cells with no neighbors
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, valid)
  
  # --- 1b. Build (id, year) -> row_idx lookup via keyed join -----------------
  idx_table <- dt[, .(id, year, .row_idx)]
  setkey(idx_table, id, year)
  
  # --- 1c. For each row, find its neighbors' row indices ---------------------
  #     Strategy: join cell_data rows with edges on id = from_id,
  #     then join back to idx_table to get neighbor row indices.
  
  # Get (from_id, year, source_row_idx)
  source <- dt[, .(from_id = id, year, src_row = .row_idx)]
  
  # Join: for each source row, expand to all neighbor cell IDs
  # source × edges on from_id
  setkey(source, from_id)
  setkey(edges, from_id)
  expanded <- edges[source, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: from_id, to_id, year, src_row
  
  # Join: map (to_id, year) -> neighbor row index
  setnames(idx_table, c("id", "year", ".row_idx"), c("to_id", "year", "nbr_row"))
  setkey(idx_table, to_id, year)
  setkey(expanded, to_id, year)
  expanded <- idx_table[expanded, on = c("to_id", "year"), nomatch = NA_integer_]
  # expanded now has: to_id, year, nbr_row, from_id, src_row
  
  # Drop rows where the neighbor wasn't found in the data
  expanded <- expanded[!is.na(nbr_row)]
  
  # Sort by src_row for efficient grouped operations later
  setkey(expanded, src_row)
  
  return(expanded)
  # Result columns: src_row (row in dt), nbr_row (neighbor's row in dt), year, to_id, from_id
}

cat("Building neighbor lookup...\n")
system.time({
  neighbor_edges <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~20-60 seconds

# =============================================================================
# STEP 2: COMPUTE NEIGHBOR STATS (vectorized via data.table grouped ops)
# =============================================================================
compute_and_add_all_neighbor_features <- function(dt, neighbor_edges, var_names) {
  # dt             : main data.table (with .row_idx)
  # neighbor_edges : data.table with (src_row, nbr_row) from Step 1
  # var_names      : character vector of variable names to compute neighbor stats for
  
  n_rows <- nrow(dt)
  
  for (vname in var_names) {
    cat("  Processing neighbor stats for:", vname, "\n")
    
    # Extract the variable values for all neighbor rows
    # (vectorized indexing — one shot)
    neighbor_edges[, nbr_val := dt[[vname]][nbr_row]]
    
    # Grouped aggregation: max, min, mean per source row, excluding NAs
    stats <- neighbor_edges[!is.na(nbr_val),
                            .(nbr_max  = max(nbr_val),
                              nbr_min  = min(nbr_val),
                              nbr_mean = mean(nbr_val)),
                            keyby = src_row]
    
    # Create full-length result vectors (default NA for rows with no valid neighbors)
    col_max  <- rep(NA_real_, n_rows)
    col_min  <- rep(NA_real_, n_rows)
    col_mean <- rep(NA_real_, n_rows)
    
    col_max[stats$src_row]  <- stats$nbr_max
    col_min[stats$src_row]  <- stats$nbr_min
    col_mean[stats$src_row] <- stats$nbr_mean
    
    # In-place assignment (no copy triggered)
    max_name  <- paste0(vname, "_max")
    min_name  <- paste0(vname, "_min")
    mean_name <- paste0(vname, "_mean")
    
    set(dt, j = max_name,  value = col_max)
    set(dt, j = min_name,  value = col_min)
    set(dt, j = mean_name, value = col_mean)
    
    rm(stats, col_max, col_min, col_mean)
  }
  
  # Clean up temporary column
  neighbor_edges[, nbr_val := NULL]
  
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, neighbor_edges, neighbor_source_vars)
})
# Expected: ~1-5 minutes total for all 5 variables

# Free the edge table
rm(neighbor_edges)
gc()

# =============================================================================
# STEP 3: RANDOM FOREST PREDICTION (chunked, memory-safe)
# =============================================================================
predict_chunked <- function(model, dt, feature_names, chunk_size = 500000L) {
  # model         : pre-trained RF model (randomForest or ranger object)
  # dt            : data.table containing all feature columns
  # feature_names : character vector of the ~110 predictor column names
  # chunk_size    : rows per prediction chunk (tune to RAM; 500K ≈ 440 MB per chunk)
  
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  # Detect model type
  is_ranger <- inherits(model, "ranger")
  
  cat(sprintf("Predicting %d rows in %d chunks of up to %d...\n", n, n_chunks, chunk_size))
  
  for (chunk_i in seq_len(n_chunks)) {
    start_row <- (chunk_i - 1L) * chunk_size + 1L
    end_row   <- min(chunk_i * chunk_size, n)
    
    # Extract chunk as a plain data.frame (most predict methods expect this)
    chunk_dt <- dt[start_row:end_row, ..feature_names]
    
    if (is_ranger) {
      # ranger: multithreaded prediction
      pred <- predict(model, data = chunk_dt, num.threads = parallel::detectCores())$predictions
    } else {
      # randomForest or other
      pred <- predict(model, newdata = chunk_dt)
    }
    
    predictions[start_row:end_row] <- pred
    
    rm(chunk_dt, pred)
    if (chunk_i %% 3 == 0) gc()  # periodic gc every 3 chunks
    
    if (chunk_i %% 5 == 0 || chunk_i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %d-%d)\n", chunk_i, n_chunks, start_row, end_row))
    }
  }
  
  return(predictions)
}

# ---- Load model once --------------------------------------------------------
cat("Loading trained RF model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")  # <-- adjust path

# ---- Get feature names (must match training) --------------------------------
# Option A: if stored with the model
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores the names used at training time
  feature_names <- rownames(rf_model$importance)
} else {
  # Fallback: specify manually
  # feature_names <- c("ntl", "ec", ..., "pop_density_mean")
  stop("Unknown model class: ", class(rf_model), ". Please specify feature_names manually.")
}

# Verify all features exist in cell_data
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# ---- Predict ----------------------------------------------------------------
cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, feature_names, chunk_size = 500000L)]
})

# ---- Cleanup ----------------------------------------------------------------
cell_data[, .row_idx := NULL]  # remove helper column
gc()

cat("Done. Predictions stored in cell_data$predicted_gdp\n")
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with string key hashing | Vectorized `data.table` keyed join on edge table | ~50–200× |
| **Neighbor stats** | `lapply` + `do.call(rbind, 6.46M-element list)` × 5 vars | `data.table` grouped aggregation (`:=`, `set()`) | ~100–500× |
| **Column assignment** | `data.frame$col <- ...` (copy-on-modify, ~5.7 GB copies) | `data.table::set()` (in-place, zero-copy) | Eliminates ~5 full copies |
| **Prediction** | Single call on 6.46M rows (OOM risk) or row-by-row loop | 500K-row chunks; auto-detects `ranger` multithreading | Fits in 16 GB RAM; ~2–10× if `ranger` |
| **Overall estimated time** | 86+ hours | **30–60 minutes** | ~100× |

The trained Random Forest model is loaded read-only and never modified. The numerical predictions are identical because the same model, same features, and same computation (max, min, mean of neighbors) are preserved — only the implementation mechanics change.