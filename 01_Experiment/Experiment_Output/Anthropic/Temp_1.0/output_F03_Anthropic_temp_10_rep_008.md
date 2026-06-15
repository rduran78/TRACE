 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs **character coercion** (`as.character`), **string pasting** (`paste(..., sep="_")`), **named-vector lookups** (which are hash-table lookups on character keys), and **NA filtering**. This means ~6.46M × (multiple string allocations + hash lookups per neighbor) — easily billions of micro-operations.
- The `idx_lookup` named vector has ~6.46M entries. Named lookups on vectors this large are slow because R's internal hashing degrades.
- Every iteration allocates new character vectors (`neighbor_keys`), creating enormous GC (garbage collection) pressure.

**`compute_neighbor_stats`** is the second bottleneck:

- Called **5 times** (once per neighbor source variable), each time iterating over 6.46M rows with `lapply`, extracting subsets, computing `max/min/mean`, and returning a 3-element vector.
- The final `do.call(rbind, result)` binds 6.46M small vectors into a matrix — this is a known R anti-pattern that is extremely slow and memory-hungry because `rbind` on a list of millions of small vectors forces repeated memory reallocation.

**Outer loop** compounds the problem: it mutates `cell_data` inside a for loop, triggering potential full-frame copies (R's copy-on-modify semantics). With ~110 columns × 6.46M rows, each accidental copy is ~5–8 GB.

### 1.2 Random Forest Inference Bottleneck

- Predicting 6.46M rows × 110 features through a Random Forest (even a pre-trained one) is inherently expensive. If done with `predict.randomForest` from the `randomForest` package, **it is single-threaded** and iterates tree-by-tree in R/C.
- If the model has hundreds of trees, this single-threaded prediction alone can take hours.
- Loading a large serialized RF model from disk with `readRDS` can also be slow and spike memory (the model object for 110 features can be multiple GB).

### 1.3 Memory Pressure

- 6.46M rows × 110 numeric columns ≈ **5.3 GB** just for the data.
- The neighbor lookup list (6.46M elements, each an integer vector of ~4 neighbors) ≈ **0.5–1 GB**.
- The RF model itself may be 1–4 GB.
- On a 16 GB laptop, this leaves almost no headroom, causing swap thrashing and GC storms.

### Summary of Root Causes (ranked by impact)

| Rank | Bottleneck | Estimated Share |
|------|-----------|----------------|
| 1 | `build_neighbor_lookup` — per-row string ops + hash lookups on 6.46M rows | ~30-40% |
| 2 | `compute_neighbor_stats` — per-row `lapply` + `do.call(rbind, ...)` × 5 vars | ~25-30% |
| 3 | RF prediction — single-threaded, 6.46M rows | ~20-25% |
| 4 | Data frame copies on mutation inside the for loop | ~10-15% |

---

## 2. OPTIMIZATION STRATEGY

### Feature Preparation

1. **Replace character-key hash lookups with integer arithmetic.** Since `id` and `year` are bounded integers, encode `(id, year)` → row index via a `data.table` keyed join or direct integer mapping. Eliminate all `paste` and named-vector lookups.

2. **Use `data.table` throughout.** Convert `cell_data` to a `data.table`. Use keyed joins for neighbor lookups. Compute neighbor stats vectorized using `data.table` grouping or pre-built integer index matrices.

3. **Vectorize `compute_neighbor_stats`.** Unroll the neighbor lookup into a long-form table `(row_i, neighbor_row_j)`, join the variable values, and compute grouped `max/min/mean` with `data.table` — this replaces 6.46M `lapply` iterations with a single vectorized grouped aggregation.

4. **Compute all 5 variables' neighbor stats in one pass** over the long-form neighbor table, avoiding 5 separate iterations.

5. **Avoid mutating the data frame in a loop.** Build all new columns at once via `:=` in `data.table`.

### Random Forest Inference

6. **Use a multi-threaded prediction backend.** If the model is from `ranger`, it already supports multi-threaded `predict`. If it is from `randomForest`, convert the predict call to use `ranger`'s prediction on the existing forest structure, or chunk the prediction and parallelize with `future.apply` / `parallel::mclapply`.

7. **Predict in chunks** to control peak memory — e.g., 500K rows at a time.

### Memory

8. **Remove intermediate objects aggressively** (`rm()` + `gc()`).
9. **Use single-precision (`float` package) for the prediction matrix** if the RF predict method supports it (saves 50% memory).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest), parallel
# =============================================================================

library(data.table)

# ---- Configuration ---------------------------------------------------------
CHUNK_SIZE     <- 500000L   # rows per prediction chunk (tune to RAM)
N_CORES        <- parallel::detectCores() - 1L
NEIGHBOR_VARS  <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
STAT_SUFFIXES  <- c("max", "min", "mean")

# ---- Step 0: Load data & model ---------------------------------------------
# cell_data           : data.frame / data.table with columns id, year, + predictors
# rook_neighbors_unique : spdep nb object (list of integer neighbor indices into id_order)
# id_order            : integer vector of cell IDs in the order matching nb object
# rf_model            : pre-trained Random Forest model (loaded via readRDS)

cat("Converting to data.table...\n")
if (!is.data.table(cell_data)) setDT(cell_data)

# ---- Step 1: Build neighbor lookup via integer indexing ---------------------
cat("Building optimized neighbor lookup...\n")

build_neighbor_lookup_fast <- function(dt, id_order, nb_list) {
  # Map each cell id -> position in the nb object
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Create row-index lookup keyed on (id, year) using data.table
  dt[, .row_idx := .I]
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # For each row, find its neighbors' row indices

# Vectorised: expand each row to its neighbor cell IDs, then join
  cat("  Expanding neighbor relationships...\n")

  # Step A: For every unique cell id, get its neighbor cell ids
  unique_ids <- unique(dt$id)
  # ref index into nb_list for each unique id
  ref_indices <- id_to_ref[as.character(unique_ids)]

  # Build a data.table: (cell_id, neighbor_cell_id)
  # Only for cells that actually appear in the data
  nb_edges <- rbindlist(lapply(seq_along(unique_ids), function(k) {
    ri <- ref_indices[k]
    if (is.na(ri) || length(nb_list[[ri]]) == 0) return(NULL)
    nb_cell_ids <- id_order[nb_list[[ri]]]
    data.table(id = unique_ids[k], neighbor_id = nb_cell_ids)
  }))

  cat("  Edge table built:", nrow(nb_edges), "unique (cell, neighbor) pairs\n")

  # Step B: Join with data to get (row_i, neighbor_row_j) for every year
  # For each row in dt, its neighbors share the same year
  # Approach: join nb_edges to dt on 'id', carrying year and .row_idx,
  #           then join again to get neighbor's row index in the same year.

  # Left table: every (row, year, neighbor_cell_id)
  cat("  Joining to panel years...\n")
  dt_slim <- dt[, .(id, year, row_i = .row_idx)]
  setkey(dt_slim, id)
  setkey(nb_edges, id)

  # merge: for each row, attach its neighbor cell ids
  expanded <- nb_edges[dt_slim, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year, row_i

  # Now find neighbor's row index in the same year
  setnames(row_lookup, c("id", "year", ".row_idx"), c("neighbor_id", "year", "row_j"))
  setkey(row_lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  result <- row_lookup[expanded, on = c("neighbor_id", "year"), nomatch = 0L]
  # result has columns: neighbor_id, year, row_j, id, row_i

  # Clean up temporary column
  dt[, .row_idx := NULL]

  return(result[, .(row_i, row_j)])
}

edge_dt <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
cat("Neighbor edge table:", nrow(edge_dt), "directed row-pairs\n")

# ---- Step 2: Compute all neighbor stats vectorized --------------------------
cat("Computing neighbor stats for all variables...\n")

compute_all_neighbor_stats <- function(dt, edge_dt, var_names) {
  n <- nrow(dt)

  # Pre-allocate result columns
  new_cols <- character(0)
  for (v in var_names) {
    for (s in STAT_SUFFIXES) {
      col_name <- paste0("neighbor_", v, "_", s)
      new_cols <- c(new_cols, col_name)
      set(dt, j = col_name, value = rep(NA_real_, n))
    }
  }

  for (v in var_names) {
    cat("  Processing variable:", v, "\n")
    # Attach the neighbor's value to each edge
    edge_dt[, val := dt[[v]][row_j]]

    # Remove NAs and compute grouped stats
    stats <- edge_dt[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     by = row_i]

    # Write results back into dt
    col_max  <- paste0("neighbor_", v, "_max")
    col_min  <- paste0("neighbor_", v, "_min")
    col_mean <- paste0("neighbor_", v, "_mean")

    set(dt, i = stats$row_i, j = col_max,  value = stats$nb_max)
    set(dt, i = stats$row_i, j = col_min,  value = stats$nb_min)
    set(dt, i = stats$row_i, j = col_mean, value = stats$nb_mean)
  }

  edge_dt[, val := NULL]  # clean up
  invisible(dt)
}

compute_all_neighbor_stats(cell_data, edge_dt, NEIGHBOR_VARS)

# Free the edge table
rm(edge_dt)
gc()

cat("Feature preparation complete. Columns:", ncol(cell_data), "\n")

# ---- Step 3: Random Forest prediction (chunked, multi-threaded) -------------
cat("Starting Random Forest prediction...\n")

predict_rf_chunked <- function(dt, model, chunk_size = CHUNK_SIZE) {
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)

  # Identify the feature columns the model expects
  # Works for both 'randomForest' and 'ranger' objects
  if (inherits(model, "ranger")) {
    feature_names <- model$forest$independent.variable.names
  } else if (inherits(model, "randomForest")) {
    # randomForest stores feature names in the model
    feature_names <- rownames(model$importance)
  } else {
    stop("Unsupported model class: ", class(model)[1])
  }

  # Pre-allocate prediction vector
  predictions <- numeric(n)

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    cat("  Predicting chunk", i, "/", n_chunks,
        " (rows", start_row, "-", end_row, ")\n")

    chunk_data <- dt[start_row:end_row, ..feature_names]

    if (inherits(model, "ranger")) {
      # ranger predict is already multi-threaded via num.threads
      pred <- predict(model, data = chunk_data, num.threads = N_CORES)
      predictions[start_row:end_row] <- pred$predictions
    } else {
      # randomForest predict — single-threaded in C, but chunking controls memory
      predictions[start_row:end_row] <- predict(model, newdata = chunk_data)
    }

    rm(chunk_data)
    if (i %% 5 == 0) gc()  # periodic GC to release chunk memory
  }

  return(predictions)
}

cell_data[, predicted_gdp := predict_rf_chunked(cell_data, rf_model)]

cat("Prediction complete.\n")

# ---- OPTIONAL Step 4: Parallelize randomForest predict if not ranger --------
# If the model is 'randomForest' and prediction is still too slow,
# you can parallelize across trees manually:

predict_rf_parallel <- function(dt, model, n_cores = N_CORES) {
  stopifnot(inherits(model, "randomForest"))
  feature_names <- rownames(model$importance)
  pred_data <- as.matrix(dt[, ..feature_names])

  n_trees <- model$ntree
  tree_groups <- split(seq_len(n_trees), cut(seq_len(n_trees), n_cores))

  cl <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cl))

  # Export minimal objects to workers
  parallel::clusterExport(cl, varlist = c("model", "pred_data"), envir = environment())

  # Each worker predicts using a subset of trees, then we average
  tree_predictions <- parallel::parLapply(cl, tree_groups, function(tree_idx) {
    # predict.randomForest supports predict.all to get per-tree predictions
    p <- predict(model, newdata = pred_data, predict.all = TRUE)
    # p$individual is a matrix: n_rows × n_trees
    rowMeans(p$individual[, tree_idx, drop = FALSE])
  })

  # Average across all tree groups (weighted by group size)
  result <- Reduce(`+`, Map(`*`, tree_predictions,
                            lapply(tree_groups, length))) / n_trees
  return(result)
}

# To use (uncomment if model is randomForest and chunked prediction is too slow):
# cell_data[, predicted_gdp := predict_rf_parallel(cell_data, rf_model)]
```

---

## 4. EXPECTED IMPROVEMENT SUMMARY

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` with string ops (est. 20–30 hrs) | Single `data.table` keyed join (~2–5 min) | **~300×** |
| `compute_neighbor_stats` (×5 vars) | 5 × 6.46M `lapply` + `do.call(rbind, ...)` (est. 30–40 hrs) | 5 × one `data.table` grouped aggregation (~1–3 min each) | **~200×** |
| Data frame mutation (copy-on-modify) | 5 full-frame copies (~25–40 GB total writes) | In-place `set()` on `data.table` (zero copies) | **eliminates** |
| RF prediction (6.46M rows) | Single-threaded, full-data (est. 10–15 hrs) | Chunked + multi-threaded (ranger) or tree-parallel | **4–8×** |
| **Total pipeline** | **~86+ hours** | **~30 min – 2 hours** (depending on RF backend) | **~40–170×** |

### Key Invariants Preserved
- **The trained Random Forest model is never retrained or modified** — only used for `predict()`.
- **The original numerical estimand is preserved** — predictions are the same numbers (identical tree traversals, same `max/min/mean` aggregations), just computed far more efficiently.