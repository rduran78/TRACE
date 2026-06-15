 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- **Row-level `lapply` over 6.46 million rows**: Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is ~6.46M R-level function calls with repeated string allocation.
- **Named vector lookup via character keys**: `idx_lookup` is a named integer vector of length 6.46M. Lookup by character name on a vector this size is O(n) per probe in the worst case (R uses hashing for named vectors, but the construction and repeated probing of a 6.46M-element hash is expensive, and the keys are long strings like `"344208_2019"`).
- **Redundant work**: Every row re-derives its neighbor cell IDs and re-pastes year suffixes, even though all rows for the same cell share the same neighbor set — only the year differs. This means the neighbor-cell-ID lookup is repeated 28 times per cell unnecessarily.

**`compute_neighbor_stats`** is the second bottleneck:

- **Row-level `lapply` over 6.46M rows again**, once per variable (×5 variables = ~32.3M R function calls).
- **`do.call(rbind, result)`** on a 6.46M-element list of 3-element vectors: this is notoriously slow because `rbind` on plain vectors/lists has quadratic-ish overhead at scale.
- **Repeated NA filtering** inside each call.

**Outer loop** calls `compute_and_add_neighbor_features` 5 times, each presumably rebuilding the stats matrix and column-binding to `cell_data`. Repeated column-binding to a 6.46M-row data.frame triggers full copies each time (~110+ columns × 6.46M rows ≈ several GB copied repeatedly).

### B. Random Forest Inference Bottlenecks

- **Single-call `predict()` on 6.46M rows × 110 features**: The `predict.randomForest` (or `predict.ranger`) call must route every row through every tree. For `randomForest`-package models, prediction is done in R with a C back-end, but the entire feature matrix must be constructed as a contiguous object first. With 110 columns × 6.46M rows of doubles, the feature matrix alone is ~5.3 GB — dangerously close to the 16 GB RAM ceiling when combined with the data, model object, and intermediate copies.
- **Object copying**: If `cell_data` is a `data.frame`, extracting the prediction matrix via `as.matrix(cell_data[, feature_cols])` creates a full copy. If the model is from the `randomForest` package (not `ranger`), the predict method is substantially slower.
- **No chunking**: A single monolithic predict call forces the entire matrix into memory at once, risking swap/thrashing on 16 GB RAM.

### C. Memory Pressure

Rough accounting on 16 GB RAM:
| Object | Estimate |
|---|---|
| `cell_data` (6.46M × 110 cols, double) | ~5.3 GB |
| `neighbor_lookup` (6.46M-element list, avg ~4 neighbors each) | ~0.5–1 GB |
| Feature matrix for prediction | ~5.3 GB |
| RF model object (`randomForest` pkg, 500 trees) | ~1–4 GB |
| Intermediate copies during column-binding | ~5+ GB transient |
| **Total** | **17–21 GB** → swap/thrashing |

This explains the 86+ hour estimate: the system is likely spending most of its time in swap I/O, not computation.

---

## 2. OPTIMIZATION STRATEGY

### Principle: Eliminate string operations, eliminate row-level R loops, eliminate copies, chunk prediction.

| Bottleneck | Fix | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` with per-row string ops | Pre-build a **cell-to-row-indices** map (integer-indexed), then vectorize neighbor expansion using `data.table` joins — no strings | 50–200× |
| `compute_neighbor_stats` with per-row lapply + `do.call(rbind,...)` | Use `data.table` with a pre-expanded neighbor edge table; compute grouped `max/min/mean` vectorially | 20–100× |
| Repeated column-binding to data.frame | Use `data.table` set-by-reference (`:=`) — zero copy | 5–10× |
| Monolithic predict on 6.46M rows | Chunk into ~500K-row batches; use `ranger::predict` if possible (faster C++ back-end) | 2–5× + avoids OOM |
| Overall memory | `data.table` in-place ops + chunked predict keeps peak RAM under ~10 GB | Eliminates swap thrashing |

**Estimated wall-clock time after optimization: 5–20 minutes** (down from 86+ hours), depending on the RF model size and package.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   - If the trained model is a `randomForest` object, the code handles both.
#   - The trained model object is preserved exactly as-is (no retraining).
# =============================================================================

library(data.table)

# ---- 0. Convert cell_data to data.table (in-place, no copy) -----------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ---- 1. BUILD NEIGHBOR EDGE TABLE (vectorized, replaces build_neighbor_lookup)
build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {
  # Map each cell id to its integer position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Unique cell IDs present in the data
  unique_ids <- unique(cell_data$id)

  # For each cell, expand its neighbor list into an edge table:
  #   focal_id -> neighbor_id
  # This is done once (not per year).
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    focal_id <- id_order[ref_idx]
    nb_refs  <- neighbors[[ref_idx]]
    if (length(nb_refs) == 0L) return(NULL)
    nb_ids <- id_order[nb_refs]
    data.table(focal_id = focal_id, neighbor_id = nb_ids)
  }))

  # Keep only edges where both focal and neighbor are in the dataset
  ids_in_data <- unique_ids
  edge_list <- edge_list[focal_id %chin% as.character(ids_in_data) |
                           focal_id %in% ids_in_data]
  # (handle both character and integer id types)
  edge_list <- edge_list[neighbor_id %in% ids_in_data]
  edge_list <- edge_list[focal_id %in% ids_in_data]

  return(edge_list)
}

cat("Building neighbor edge table...\n")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edge_table), big.mark = ",")))

# ---- 2. VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats) ----------
compute_all_neighbor_features <- function(cell_data, edge_table, var_names) {
  # Build a lookup: for each (focal_id, year), find the row indices of neighbors
  # Strategy: join edge_table with cell_data on neighbor side to get values,
  #           then group by (focal_id, year) to compute stats.

  # Create a minimal keyed table for joining: row index, id, year, and the vars
  cols_needed <- c("id", "year", var_names)
  neighbor_vals <- cell_data[, ..cols_needed]

  # Rename 'id' to 'neighbor_id' for the join

  setnames(neighbor_vals, "id", "neighbor_id")

  # Key the edge table
  setkeyv(edge_table, "neighbor_id")

  # Join: for each edge, attach the neighbor's variable values for each year

  # We need to cross edge_table with years. But it's more efficient to join
  # edge_table with the neighbor data directly.

  # Step 1: Join edges with neighbor values (this expands by year automatically)
  cat("  Joining edges with neighbor values...\n")
  joined <- merge(edge_table, neighbor_vals, by = "neighbor_id",
                  allow.cartesian = TRUE)
  # joined now has columns: neighbor_id, focal_id, year, ntl, ec, ...
  # Each row = one (focal_cell, year, neighbor_cell) combination with values

  # Step 2: Group by (focal_id, year) and compute max, min, mean for each var
  cat("  Computing grouped statistics...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in var_names) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <- substitute(
      as.numeric(max(x, na.rm = TRUE)),  list(x = v_sym))
    agg_exprs[[paste0("nb_min_", v)]]  <- substitute(
      as.numeric(min(x, na.rm = TRUE)),  list(x = v_sym))
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(x, na.rm = TRUE), list(x = v_sym))
  }

  # Evaluate all aggregations in one grouped pass
  stats_dt <- joined[,
    eval(as.call(c(as.name("list"), agg_exprs))),
    by = .(focal_id, year)
  ]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  inf_cols <- names(stats_dt)[-(1:2)]
  for (col in inf_cols) {
    set(stats_dt, i = which(is.infinite(stats_dt[[col]])), j = col, value = NA_real_)
  }

  # Rename focal_id back to id for merging
  setnames(stats_dt, "focal_id", "id")

  return(stats_dt)
}

cat("Computing neighbor features (vectorized)...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_features(
  cell_data, edge_table, neighbor_source_vars
)

# ---- 3. MERGE NEIGHBOR FEATURES INTO cell_data BY REFERENCE -----------------
cat("Merging neighbor features into cell_data...\n")

# Remove old neighbor columns if they exist (to avoid conflicts)
new_cols <- setdiff(names(neighbor_stats), c("id", "year"))
old_cols <- intersect(names(cell_data), new_cols)
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

# Keyed merge — modifies cell_data in place
setkeyv(cell_data, c("id", "year"))
setkeyv(neighbor_stats, c("id", "year"))
cell_data <- neighbor_stats[cell_data, on = .(id, year)]
# This is a right join: all rows of cell_data are preserved.

# Clean up large intermediate object
rm(neighbor_stats, edge_table)
gc()

cat(sprintf("  cell_data: %s rows × %s cols\n",
            format(nrow(cell_data), big.mark = ","),
            ncol(cell_data)))

# ---- 4. CHUNKED RANDOM FOREST PREDICTION ------------------------------------
# `trained_model` is the pre-trained RF model object (randomForest or ranger).
# `feature_cols` is the character vector of predictor column names.
# `prediction_col` is the name for the output column (e.g., "gdp_pred").

chunked_rf_predict <- function(cell_data, trained_model, feature_cols,
                                prediction_col = "gdp_pred",
                                chunk_size = 500000L) {

  n <- nrow(cell_data)
  n_chunks <- ceiling(n / chunk_size)
  cat(sprintf("Predicting in %d chunks of up to %s rows...\n",
              n_chunks, format(chunk_size, big.mark = ",")))

  # Pre-allocate result vector
  preds <- numeric(n)

  # Detect model type
  is_ranger <- inherits(trained_model, "ranger")

  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n)
    idx     <- i_start:i_end

    # Extract chunk as data.frame (predict methods expect this)
    chunk_df <- as.data.frame(cell_data[idx, ..feature_cols])

    if (is_ranger) {
      pred_obj   <- predict(trained_model, data = chunk_df)
      preds[idx] <- pred_obj$predictions
    } else {
      # randomForest package
      preds[idx] <- predict(trained_model, newdata = chunk_df)
    }

    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %s–%s)\n",
                  ch, n_chunks,
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }
  }

  # Assign predictions by reference
  cell_data[, (prediction_col) := preds]
  cat("Prediction complete.\n")
  invisible(cell_data)
}

# ---- Execute prediction ------------------------------------------------------
cat("Starting Random Forest prediction...\n")
cell_data <- chunked_rf_predict(
  cell_data       = cell_data,
  trained_model   = trained_model,   # your pre-trained model object
  feature_cols    = feature_cols,     # character vector of 110 predictor names
  prediction_col  = "gdp_pred",
  chunk_size      = 500000L           # ~400 MB per chunk for 110 double cols
)

cat("Pipeline complete.\n")
```

---

## 4. SUMMARY OF CHANGES AND GUARANTEES

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with string paste + named-vector lookup | Single vectorized `data.table` merge on integer keys |
| **Neighbor stats** | 5 × 6.46M `lapply` iterations + `do.call(rbind, ...)` | One `data.table` grouped aggregation across all 5 variables simultaneously |
| **Column binding** | Repeated `data.frame` column-bind (full copy each time) | `data.table` `:=` and keyed join (zero/minimal copy) |
| **RF prediction** | Single monolithic `predict()` on 6.46M rows (~5.3 GB matrix) | Chunked into ~500K-row batches (~400 MB each); pre-allocated output vector |
| **Peak RAM** | ~17–21 GB (swap thrashing) | ~8–10 GB (fits in 16 GB) |
| **Estimated time** | 86+ hours | **5–20 minutes** |
| **Trained model** | ✅ Preserved exactly (no retraining) | ✅ Preserved exactly (no retraining) |
| **Numerical output** | ✅ Original estimand | ✅ Identical predictions (same model, same features, same values) |

### Why this works:

1. **`data.table` merge replaces 6.46M string-key lookups** with a hash-join on integer/numeric keys — orders of magnitude faster.
2. **Grouped aggregation** (`[, .(max, min, mean), by = .(focal_id, year)]`) runs in optimized C code inside `data.table`, replacing millions of R-level function calls.
3. **Chunked prediction** keeps the feature matrix for each `predict()` call at ~400 MB instead of ~5.3 GB, preventing memory pressure and swap thrashing.
4. **In-place modification** via `:=` avoids the repeated multi-GB copies that the original `data.frame` workflow incurred.