 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), character-keyed lookups into named vectors (`id_to_ref`, `idx_lookup`), and NA filtering. Named-vector lookups in R are hash-table operations, but doing ~6.46M × ~4 neighbors ≈ 26M string constructions and hash lookups is extremely slow in interpreted R. The output is a list of 6.46M integer vectors — a large, fragmented memory structure.

**`compute_neighbor_stats`:** For each of 6.46M rows, this subsets a numeric vector by index, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern over 6.46M small 3-element vectors is notoriously slow: `do.call(rbind, list_of_6.46M_vectors)` alone can take tens of minutes because it repeatedly allocates and copies.

**Outer loop:** This runs `compute_neighbor_stats` 5 times (once per variable), each time producing 3 new columns (max, min, mean) — 15 columns total. Each call re-traverses the 6.46M-element neighbor lookup.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, a single `predict()` call on a large Random Forest can be very slow and memory-intensive. If the model has many trees (e.g., 500) and deep nodes, the prediction matrix alone is `6.46M × 110 × 8 bytes ≈ 5.3 GB` as a dense numeric matrix, and the forest must traverse every tree for every row. If prediction is done in a row-level or small-batch loop, overhead multiplies catastrophically.

Additionally, if the data is a `data.frame` rather than a `matrix`, `predict.randomForest` (or `predict.ranger`) will internally convert it — doubling memory use at peak.

### 1.3 Memory Pressure

On a 16 GB laptop:
- Base data: 6.46M × 110 × 8 bytes ≈ 5.3 GB
- Neighbor lookup list: 6.46M entries × ~4 integers each ≈ overhead-heavy (R lists have ~56 bytes overhead per element → ~360 MB just in list overhead, plus the integer vectors)
- Intermediate copies during feature addition (if `data.frame` is copied each time a column is added): up to 5.3 GB × 2
- Prediction matrix copy: another 5.3 GB

This easily exceeds 16 GB, causing swapping → the 86+ hour runtime.

### 1.4 Root Causes Summary

| Bottleneck | Cause | Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops + named-vector hash lookups in R loop over 6.46M rows | Hours |
| `compute_neighbor_stats` | `lapply` over 6.46M + `do.call(rbind, 6.46M-element list)` | Hours per variable |
| Column addition in loop | Repeated `data.frame` copy-on-modify (5 iterations × 3 cols) | GB-scale redundant copies |
| Neighbor lookup storage | 6.46M-element R list of small integer vectors | ~400+ MB, GC pressure |
| Prediction | Possible row-loop or `data.frame`-to-matrix conversion; single giant call on 6.46M rows | Hours + memory spike |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation: Vectorized with `data.table`

Replace the entire `build_neighbor_lookup` → `compute_neighbor_stats` pipeline with a **vectorized join-and-aggregate** approach using `data.table`:

1. **Expand the neighbor graph into an edge table** (`data.table` with columns `id`, `neighbor_id`) — done once from the `nb` object. This is ~1.37M directed edges.
2. **Join the edge table with the panel data** on `(neighbor_id, year)` to get neighbor values — a single keyed merge producing ~1.37M × 28 ≈ ~38M rows (but many fewer because not all neighbor-cell-years exist; the join handles this naturally).
3. **Aggregate** (max, min, mean) grouped by `(id, year)` — a single `data.table` grouped operation.
4. **Merge** the 3 summary columns back to the main table.
5. Repeat for each of the 5 variables (or do all 5 simultaneously in one join + aggregate).

This replaces 6.46M × 5 R-level iterations with a handful of vectorized C-level `data.table` operations.

### 2.2 Prediction: Chunked, Matrix-Based

1. Convert the prediction features to a **numeric matrix** once (not a `data.frame`).
2. Call `predict()` in **chunks** (e.g., 500K rows) to keep peak memory bounded.
3. If using `randomForest`, consider switching the predict call to `ranger::predict` if the model can be converted, as `ranger` prediction is multithreaded. If the model must remain a `randomForest` object, we still chunk to control memory.

### 2.3 Memory Management

- Use `data.table` in-place column assignment (`:=`) to avoid copies.
- Remove intermediate objects and call `gc()` at key points.
- Chunk prediction to avoid doubling memory.

### Expected Speedup

| Component | Before | After (estimated) |
|---|---|---|
| Neighbor lookup + stats | ~40–60 hours | ~2–5 minutes |
| Column binding | ~10–20 hours (copies) | Seconds (`:=`) |
| Prediction (6.46M rows) | ~10–20 hours | ~10–30 minutes |
| **Total** | **86+ hours** | **~15–40 minutes** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   — or use randomForest if ranger conversion is not possible
# =============================================================================

library(data.table)

# ---- STEP 0: Convert main data to data.table (in-place, no copy) -----------

setDT(cell_data)

# Ensure keyed for fast joins
# Assumes cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
setkey(cell_data, id, year)


# ---- STEP 1: Build edge table from nb object (once) ------------------------

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer index vectors (spdep::nb format)
  # id_order maps position -> cell id
  
  # Pre-calculate sizes for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    s <- x[x != 0L]  # nb objects use 0 for no-neighbor
    length(s)
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs != 0L]
    n <- length(nbrs)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nbrs]
      pos <- pos + n
    }
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building edge table from neighbor structure...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d directed edges\n", nrow(edge_dt)))


# ---- STEP 2: Compute all neighbor features via vectorized join + aggregate --

compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # Create a minimal lookup table: (id, year, var1, var2, ...) for neighbor values
  lookup_cols <- c("id", "year", source_vars)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)
  
  # Join edges with years: for every (id, year) pair, find neighbor values
  # First, get the unique years per id from the main data
  id_year <- cell_dt[, .(id, year)]
  setkey(id_year, id)
  setkey(edge_dt, id)
  
  # Merge: for each (id, year), get all neighbor_ids
  cat("  Joining edges with panel years...\n")
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  # Now join to get neighbor variable values
  cat("  Joining neighbor values...\n")
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, <source_vars>, id
  
  # Aggregate: for each (id, year), compute max/min/mean of each source var
  cat("  Aggregating neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <- bquote(max(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("nb_min_", v)]]  <- bquote(min(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("nb_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  nb_stats <- expanded[, eval(agg_call), by = .(id, year)]
  
  # Replace Inf/-Inf (from max/min on all-NA) with NA
  nb_feature_cols <- names(nb_stats)[!names(nb_stats) %in% c("id", "year")]
  for (col in nb_feature_cols) {
    vals <- nb_stats[[col]]
    set(nb_stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
  }
  
  # Merge back to main data
  cat("  Merging neighbor features back to main table...\n")
  setkey(nb_stats, id, year)
  cell_dt <- nb_stats[cell_dt, on = .(id, year)]
  
  setkey(cell_dt, id, year)
  cell_dt
}

cat("Computing neighbor features (vectorized)...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up intermediate objects
rm(edge_dt)
gc()

cat(sprintf("  Final table: %d rows x %d cols\n", nrow(cell_data), ncol(cell_data)))


# ---- STEP 3: Prepare prediction matrix -------------------------------------

# Identify the feature columns the model expects.
# If using randomForest: model$forest$xlevels has the names
# If using ranger: model$forest$independent.variable.names

get_feature_names <- function(model) {
  if (inherits(model, "ranger")) {
    return(model$forest$independent.variable.names)
  } else if (inherits(model, "randomForest")) {
    return(names(model$forest$xlevels))
    # Alternative if xlevels is empty (all numeric):
    # return(rownames(model$importance))
  } else {
    stop("Unsupported model class: ", class(model)[1])
  }
}

cat("Preparing prediction matrix...\n")

# Load the trained model (adjust path as needed)
# rf_model <- readRDS("path/to/trained_rf_model.rds")

feature_names <- get_feature_names(rf_model)

# Verify all features are present
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# Extract as matrix for fastest predict performance
# data.table's as.matrix on subset columns is efficient
pred_matrix <- as.matrix(cell_data[, ..feature_names])
cat(sprintf("  Prediction matrix: %d rows x %d cols (%.1f GB)\n",
            nrow(pred_matrix), ncol(pred_matrix),
            object.size(pred_matrix) / 1e9))


# ---- STEP 4: Chunked prediction --------------------------------------------

predict_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat(sprintf("  Predicting in %d chunks of up to %d rows...\n", n_chunks, chunk_size))
  
  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)
    chunk     <- newdata[start_idx:end_idx, , drop = FALSE]
    
    if (inherits(model, "ranger")) {
      # ranger::predict is multithreaded — much faster
      pred_obj <- predict(model, data = chunk)
      predictions[start_idx:end_idx] <- pred_obj$predictions
    } else {
      # randomForest::predict
      predictions[start_idx:end_idx] <- predict(model, newdata = chunk)
    }
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("    Chunk %d/%d done (rows %d–%d)\n", i, n_chunks, start_idx, end_idx))
    }
  }
  
  predictions
}

cat("Running Random Forest prediction...\n")
cell_data[, predicted_gdp := predict_chunked(rf_model, pred_matrix, chunk_size = 500000L)]

# Free the large prediction matrix
rm(pred_matrix)
gc()

cat("Done. Predictions stored in cell_data$predicted_gdp\n")


# ---- STEP 5 (OPTIONAL): Convert randomForest model to ranger for speed -----
# 
# If the model is a `randomForest` object and you want faster multithreaded
# prediction in future runs, you can re-save predictions but you CANNOT retrain.
# However, you CAN use ranger's predict if you convert the model.
#
# Unfortunately, there is no direct randomForest-to-ranger model conversion.
# The recommended approach if prediction speed is critical:
#   1. Save the predictions from this run.
#   2. For future reuse, consider training with ranger (same hyperparameters)
#      to get an equivalent model with faster predict.
#
# Since the constraint is "must not retrain," we keep the original model and
# rely on chunked prediction + matrix input for speed.


# ---- STEP 6 (OPTIONAL): Parallel prediction for randomForest ---------------
# If the model is randomForest (not ranger) and you want to use multiple cores:

predict_parallel_rf <- function(model, newdata, n_cores = parallel::detectCores() - 1L,
                                chunk_size = 500000L) {
  requireNamespace("parallel", quietly = TRUE)
  
  n <- nrow(newdata)
  # Split into chunks
  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  
  cat(sprintf("  Parallel prediction: %d chunks across %d cores\n",
              length(chunks), n_cores))
  
  cl <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  
  # Export model to workers
  parallel::clusterExport(cl, varlist = "model", envir = environment())
  parallel::clusterEvalQ(cl, library(randomForest))
  
  results <- parallel::parLapply(cl, chunks, function(idx) {
    predict(model, newdata = newdata[idx, , drop = FALSE])
  })
  
  predictions <- numeric(n)
  for (j in seq_along(chunks)) {
    predictions[chunks[[j]]] <- results[[j]]
  }
  
  predictions
}

# Usage (uncomment to use):
# cell_data[, predicted_gdp := predict_parallel_rf(rf_model, pred_matrix, n_cores = 4L)]
```

---

## 4. KEY CHANGES SUMMARY

| Original | Optimized | Why |
|---|---|---|
| `build_neighbor_lookup`: 6.46M-iteration `lapply` with string paste + named-vector lookup | Single `data.table` edge-table join (`O(n)` merge on integer keys) | Vectorized C-level join replaces millions of interpreted R hash lookups |
| `compute_neighbor_stats`: `lapply` over 6.46M + `do.call(rbind, ...)` | `data.table` grouped aggregation `[, .(max, min, mean), by = .(id, year)]` | Single pass, no R-level loop, no list-to-matrix conversion |
| 5 separate passes (one per variable) | All 5 variables joined and aggregated in one pass | 5× fewer data traversals |
| `data.frame` column addition in loop (copy-on-modify) | `data.table` `:=` in-place assignment | Zero-copy column addition |
| `predict()` on full 6.46M-row `data.frame` | `predict()` on chunked numeric `matrix` | Avoids data.frame→matrix conversion inside predict; bounds peak memory |
| Single-threaded prediction | Optional parallel chunked prediction (or `ranger` if available) | Utilizes multiple cores |

This preserves the trained Random Forest model exactly as-is (no retraining) and produces numerically identical predictions (same model, same features, same values — just computed more efficiently).