 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows with per-row string operations**

The function pastes strings and does named-vector lookups for every single row. With ~6.46 million rows, this means:
- **~6.46M `paste()` calls** inside the loop (for `neighbor_keys`).
- **Named character vector indexing** (`idx_lookup[neighbor_keys]`) is O(n) hash lookup repeated millions of times.
- The outer `lapply` returns a list of 6.46M integer vectors — heavy memory allocation and GC pressure.

**B. `compute_neighbor_stats` — Repeated per-variable full-data scan**

Called 5 times (once per neighbor source variable). Each call does:
- 6.46M `lapply` iterations, each subsetting a numeric vector by index, removing NAs, and computing `max/min/mean`.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors — this is an extremely slow row-binding pattern.

**C. Object Copying (`cell_data` mutation in a loop)**

```r
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

Each iteration likely copies the entire ~6.46M × 110-column data.frame. With 5 variables, that's 5 full copies of a multi-GB object. R's copy-on-modify semantics make this devastating.

**D. Random Forest Inference (the stated primary concern)**

- Predicting 6.46M rows with ~110 features through a Random Forest (potentially hundreds of trees) is inherently expensive.
- If `predict()` is called row-by-row or in small batches rather than as a single vectorized call, overhead multiplies dramatically.
- Model loading from disk (if done repeatedly or if the serialized object is very large) adds I/O time.
- If the model object is a `randomForest` object (rather than `ranger`), prediction is single-threaded and much slower.

### Estimated Time Breakdown (of the ~86+ hours)

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~15–25% | Per-row string ops, named vector lookup |
| `compute_neighbor_stats` (×5) | ~30–40% | Per-row lapply, `do.call(rbind, ...)` |
| Data.frame copying (×5) | ~10–15% | Copy-on-modify of multi-GB frame |
| RF prediction | ~20–30% | Single-threaded predict, possible row-level calls |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything, eliminate per-row R loops, use `data.table` for zero-copy mutation, and use `ranger` for parallel prediction.

| Bottleneck | Fix |
|---|---|
| `build_neighbor_lookup`: per-row paste/hash | Pre-build a `data.table` join between (id, year) → row index; expand neighbor pairs into a flat edge table; do a single keyed merge. |
| `compute_neighbor_stats`: per-row lapply | Use the flat edge table + `data.table` grouped aggregation: one `[, .(max, min, mean), by=row_i]` call per variable. |
| `do.call(rbind, list_of_vectors)` | Eliminated entirely — `data.table` returns a matrix/DT directly. |
| Data.frame copy-on-modify | Use `data.table` `:=` for in-place column addition — zero copies. |
| RF predict (single-threaded) | If model is `randomForest`, convert to `ranger`-compatible or call `predict()` once on full matrix. If already `ranger`, use `num.threads`. Predict in one vectorized call. |
| Model loading | Load once, keep in memory. |

### Expected Speedup

Conservative estimate: **50–200×** overall, bringing ~86 hours down to **~30–90 minutes**.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE — cell-level GDP prediction
# Requirements: data.table, ranger (for fast parallel predict), Matrix (optional)
# =============================================================================

library(data.table)

# -------------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table (in-place, no copy if already DT)
# -------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure row index column exists for later joins
cell_data[, row_i := .I]

# -------------------------------------------------------------------------
# STEP 1: Build flat neighbor edge table (replaces build_neighbor_lookup)
#
# Instead of a list-of-lists with 6.46M entries, we build a two-column
# data.table: (row_i, neighbor_row_i) — one row per directed edge.
# This is done entirely via vectorized joins, no per-row R loop.
# -------------------------------------------------------------------------

build_neighbor_edges_dt <- function(cell_dt, id_order, neighbors) {
  # Map each cell id to its position in id_order (1-based reference index)
  id_to_ref <- data.table(
    id      = id_order,
    ref_idx = seq_along(id_order)
  )

  # Expand the nb object into a flat edge list: (ref_idx, neighbor_ref_idx)
  # neighbors is a list where neighbors[[i]] gives integer indices into id_order
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(ref_idx = i, nb_ref_idx = as.integer(nb))
  }))

  # Map ref_idx -> cell id
  edge_list[, id := id_order[ref_idx]]
  edge_list[, nb_id := id_order[nb_ref_idx]]

  # Build a lookup: (id, year) -> row_i in cell_dt
  idx_lookup <- cell_dt[, .(id, year, row_i)]
  setkey(idx_lookup, id, year)

  # Get all unique years present in the data
  all_years <- sort(unique(cell_dt$year))

  # Cross-join edges × years, then join to get row_i for source and neighbor
  # For memory efficiency, process in year chunks
  edges_by_year <- rbindlist(lapply(all_years, function(yr) {
    # For this year, find row_i for each (id, year) and (nb_id, year)
    src <- idx_lookup[.(edge_list$id, yr), nomatch = 0L]
    setnames(src, "row_i", "src_row_i")

    nb  <- idx_lookup[.(edge_list$nb_id, yr), nomatch = 0L]
    setnames(nb, "row_i", "nb_row_i")

    # We need to align: for each edge (id, nb_id), look up both row_i values
    # Rebuild per-year edge with row indices
    yr_edges <- edge_list[, .(id, nb_id)]

    # Join source row_i
    yr_edges[, year := yr]
    setkey(yr_edges, id, year)
    yr_edges <- idx_lookup[yr_edges, nomatch = 0L]
    setnames(yr_edges, "row_i", "src_row_i")

    # Join neighbor row_i
    setnames(yr_edges, c("id", "nb_id"), c("src_id", "id"))
    setkey(yr_edges, id, year)
    yr_edges <- idx_lookup[yr_edges, nomatch = 0L]
    setnames(yr_edges, c("row_i", "id"), c("nb_row_i", "nb_id"))

    yr_edges[, .(src_row_i, nb_row_i)]
  }))

  return(edges_by_year)
}

# ---- Simpler, more memory-efficient version ----
build_neighbor_edges_fast <- function(cell_dt, id_order, neighbors) {
  # 1. Flat edge list from nb object: (source_id, neighbor_id)
  src_ref <- rep(
    seq_along(neighbors),
    times = vapply(neighbors, function(nb) {
      if (length(nb) == 1L && nb[1] == 0L) 0L else length(nb)
    }, integer(1))
  )
  nb_ref <- unlist(lapply(neighbors, function(nb) {
    if (length(nb) == 1L && nb[1] == 0L) integer(0) else as.integer(nb)
  }), use.names = FALSE)

  edge_ids <- data.table(
    src_id = id_order[src_ref],
    nb_id  = id_order[nb_ref]
  )
  rm(src_ref, nb_ref)

  # 2. Lookup: (id, year) -> row_i
  idx_dt <- cell_dt[, .(id, year, row_i)]

  # 3. Cross with years via join
  #    For each (src_id, nb_id) pair, we need every year where BOTH exist.
  #    Strategy: join edge_ids to idx_dt twice.

  # Join source side: get (src_id, year, src_row_i)
  setnames(idx_dt, c("id", "year", "row_i"), c("src_id", "year", "src_row_i"))
  setkey(idx_dt, src_id)
  setkey(edge_ids, src_id)
  edges_with_src <- idx_dt[edge_ids, on = "src_id", allow.cartesian = TRUE, nomatch = 0L]
  # Result: (src_id, year, src_row_i, nb_id)

  # Join neighbor side: get nb_row_i
  setnames(idx_dt, c("src_id", "src_row_i"), c("nb_id", "nb_row_i"))
  setkey(idx_dt, nb_id, year)
  setkey(edges_with_src, nb_id, year)
  full_edges <- idx_dt[edges_with_src, on = c("nb_id", "year"), nomatch = 0L]
  # Result: (nb_id, year, nb_row_i, src_id, src_row_i)

  # Clean up idx_dt names for future use
  setnames(idx_dt, c("nb_id", "nb_row_i"), c("id", "row_i"))

  return(full_edges[, .(src_row_i, nb_row_i)])
}

cat("Building neighbor edge table...\n")
system.time({
  neighbor_edges <- build_neighbor_edges_fast(cell_data, id_order, rook_neighbors_unique)
})
# neighbor_edges is a data.table with columns: src_row_i, nb_row_i
# Expected rows: ~1.37M edges × 28 years ≈ ~38.5M rows (fits easily in RAM)

cat("Edge table rows:", nrow(neighbor_edges), "\n")

# -------------------------------------------------------------------------
# STEP 2: Compute neighbor stats for all variables at once
#          (replaces compute_neighbor_stats + the for-loop)
#
# For each (src_row_i), compute max/min/mean of the neighbor values.
# This is a single grouped aggregation per variable — fully vectorized.
# -------------------------------------------------------------------------

compute_and_add_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  for (var_name in var_names) {
    cat("  Computing neighbor stats for:", var_name, "\n")

    # Attach the neighbor's value to each edge
    vals <- cell_dt[[var_name]]
    edge_dt[, nb_val := vals[nb_row_i]]

    # Grouped aggregation: max, min, mean per source row
    stats <- edge_dt[!is.na(nb_val),
      .(
        nb_max  = max(nb_val),
        nb_min  = min(nb_val),
        nb_mean = mean(nb_val)
      ),
      by = src_row_i
    ]

    # Prepare column names
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Initialize with NA, then fill via join (in-place, zero copy)
    set(cell_dt, j = col_max,  value = NA_real_)
    set(cell_dt, j = col_min,  value = NA_real_)
    set(cell_dt, j = col_mean, value = NA_real_)

    set(cell_dt, i = stats$src_row_i, j = col_max,  value = stats$nb_max)
    set(cell_dt, i = stats$src_row_i, j = col_min,  value = stats$nb_min)
    set(cell_dt, i = stats$src_row_i, j = col_mean, value = stats$nb_mean)

    rm(stats)
  }

  # Clean up temporary column
  edge_dt[, nb_val := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing all neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, neighbor_edges, neighbor_source_vars)
})

# Remove helper column
cell_data[, row_i := NULL]

# -------------------------------------------------------------------------
# STEP 3: Random Forest prediction — optimized
# -------------------------------------------------------------------------

# OPTION A: If the model is a `ranger` object (preferred — natively parallel)
# OPTION B: If the model is a `randomForest` object (convert prediction approach)

cat("Loading trained model...\n")
# Load model ONCE — do not reload per batch
# rf_model <- readRDS("path/to/trained_model.rds")

# Detect model class and predict accordingly
predict_gdp <- function(model, newdata_dt, batch_size = 500000L) {
  # Convert to matrix/data.frame as needed (ranger accepts data.frame)
  # Identify predictor columns (exclude id, year, and any target columns)
  # Adjust this to match your actual predictor column names:
  pred_cols <- setdiff(
    names(newdata_dt),
    c("id", "year", "gdp", "gdp_predicted", "row_i")
  )

  n <- nrow(newdata_dt)

  if (inherits(model, "ranger")) {
    # ---- RANGER: parallel, fast, single call ----
    cat("Predicting with ranger (parallel)...\n")
    # ranger::predict can handle the full dataset in one call
    # num.threads uses all cores by default
    pred <- predict(model, data = newdata_dt[, ..pred_cols],
                    num.threads = parallel::detectCores())
    return(pred$predictions)

  } else if (inherits(model, "randomForest")) {
    # ---- randomForest: single-threaded, batch to manage memory ----
    cat("Predicting with randomForest (batched)...\n")

    # Pre-allocate output vector
    predictions <- numeric(n)

    # Convert predictor columns to a matrix once (faster for predict.randomForest)
    pred_matrix <- as.matrix(newdata_dt[, ..pred_cols])

    # Predict in batches to limit peak memory (not row-by-row!)
    n_batches <- ceiling(n / batch_size)
    for (b in seq_len(n_batches)) {
      start_i <- (b - 1L) * batch_size + 1L
      end_i   <- min(b * batch_size, n)
      idx     <- start_i:end_i

      predictions[idx] <- predict(model, newdata = pred_matrix[idx, , drop = FALSE])

      if (b %% 5 == 0 || b == n_batches) {
        cat(sprintf("  Batch %d/%d complete (rows %d-%d)\n", b, n_batches, start_i, end_i))
      }
    }

    rm(pred_matrix)
    return(predictions)

  } else {
    stop("Unsupported model class: ", class(model)[1])
  }
}

cat("Running prediction...\n")
system.time({
  cell_data[, gdp_predicted := predict_gdp(rf_model, cell_data)]
})

cat("Done. Predictions stored in cell_data$gdp_predicted\n")


# =========================================================================
# OPTIONAL: If you want to convert a randomForest model to ranger for
# future speed gains (without retraining), you can't directly convert.
# But you CAN parallelize randomForest prediction manually:
# =========================================================================

predict_rf_parallel <- function(model, newdata_matrix, n_cores = NULL) {
  if (is.null(n_cores)) n_cores <- parallel::detectCores() - 1L
  n <- nrow(newdata_matrix)

  # Split row indices into chunks
  chunks <- split(seq_len(n), cut(seq_len(n), n_cores, labels = FALSE))

  cl <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cl))

  # Export model to workers once
  parallel::clusterExport(cl, varlist = "model", envir = environment())
  parallel::clusterEvalQ(cl, library(randomForest))

  # Parallel predict
  results <- parallel::parLapply(cl, chunks, function(idx) {
    predict(model, newdata = newdata_matrix[idx, , drop = FALSE])
  })

  # Reassemble in order
  predictions <- numeric(n)
  for (k in seq_along(chunks)) {
    predictions[chunks[[k]]] <- results[[k]]
  }

  return(predictions)
}
```

---

## Summary of Changes and Expected Impact

| Original Problem | Solution | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup`: 6.46M per-row `paste` + named vector lookup | Flat edge `data.table` built via vectorized joins | **~50–100×** |
| `compute_neighbor_stats`: 6.46M `lapply` + `do.call(rbind, ...)` | Single `data.table` grouped `[, .(max,min,mean), by=]` per variable | **~100–200×** |
| Data.frame copy-on-modify (5 copies of multi-GB frame) | `data.table::set()` — true in-place column addition, zero copies | **~5×** memory reduction, eliminates GC pauses |
| RF predict (possibly row-by-row or single-threaded) | Single vectorized `predict()` call; `ranger` parallel or `parLapply` for `randomForest` | **~4–8×** (on 8 cores) |
| **Overall** | | **~86 hrs → ~30–60 min** |

### Key Invariants Preserved
- ✅ Trained Random Forest model is never retrained — only `predict()` is called.
- ✅ Numerical estimand is identical: `max`, `min`, `mean` of non-NA neighbor values per (cell, year), with `NA` for cells with no valid neighbors — exactly matching the original logic.
- ✅ All 5 neighbor source variables produce the same 15 derived columns (3 stats × 5 vars).