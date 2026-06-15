You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F03_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`**: This function calls `lapply` over **~6.46 million rows**, and in each iteration it:
- Performs character coercion and named-vector lookups (`id_to_ref[as.character(...)]`),
- Pastes strings (`paste(... , sep="_")`),
- Looks up named vectors (`idx_lookup[neighbor_keys]`),
- Filters NAs.

Named vector lookup in R is **O(n)** hashing per call. Doing this 6.46M times with string construction is extremely expensive. Estimated cost: **hours**.

**`compute_neighbor_stats`**: Runs `lapply` over 6.46M entries, each extracting a variable-length subset of a numeric vector, removing NAs, computing max/min/mean, then `do.call(rbind, ...)` on 6.46M small vectors. The `do.call(rbind, list_of_vectors)` alone is a well-known R anti-pattern that copies memory repeatedly. This is called **5 times** (once per source variable), multiplying the cost.

**`compute_and_add_neighbor_features`**: Each call presumably `cbind`s or adds columns to a ~6.46M-row data.frame, which triggers a full copy of the data.frame each time.

### 1.2 Prediction Bottleneck

With ~6.46 million rows and ~110 predictors, calling `predict()` on a `ranger` or `randomForest` model in one shot can:
- Require the entire prediction matrix to be held in memory simultaneously (~5.3 GB for a dense numeric matrix),
- Be slow if the model object is a `randomForest` object (pure R tree traversal) versus `ranger` (C++ backend).

If the model is from the `randomForest` package, the `predict` method is notoriously slow on large data because it uses R-level loops over trees.

### 1.3 Summary of Root Causes

| Bottleneck | Root Cause | Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row string paste + named-vector lookup Ã— 6.46M | ~hours |
| `compute_neighbor_stats` | Per-row `lapply` + `do.call(rbind,...)` Ã— 5 vars | ~hours |
| Data.frame mutation | Copy-on-modify of 6.46M-row frame Ã— 5 iterations | ~tens of minutes, RAM spikes |
| `predict()` | Possibly `randomForest` package (R-level), full matrix in RAM | ~hours, RAM pressure |
| **Total** | | **86+ hours** |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â†’ `data.table` + Vectorized Integer Indexing

1. **Replace named-vector string lookups with integer join tables.** Build a `data.table` mapping `(id, year)` â†’ row index, and an edge list of `(row_i, neighbor_row_j)` for all cell-year pairs. This is a one-time vectorized merge.

2. **Replace per-row `lapply` with grouped aggregation.** Once we have an edge-list `(row_i, neighbor_row_j)`, computing neighbor max/min/mean is just a grouped `data.table` aggregation: `edge_dt[, .(max_v = max(v), min_v = min(v), mean_v = mean(v)), by = row_i]`. This is **fully vectorized in C** and runs in seconds.

3. **Avoid repeated data.frame copies.** Use `data.table` set-by-reference (`:=`) to add columns in-place.

### 2.2 Prediction â†’ Chunked Prediction with `ranger` Compatibility

1. If the model is `randomForest`, convert it to `ranger` format or use chunked prediction. If conversion is not feasible, predict in chunks to manage memory.
2. If the model is `ranger`, predict in chunks of ~500Kâ€“1M rows to keep peak memory under control.
3. Pre-allocate the output vector and fill by chunk index.

### 2.3 Expected Speedup

| Step | Before | After | Factor |
|---|---|---|---|
| Neighbor lookup + stats | ~60+ hours | ~2â€“5 minutes | ~1000Ã— |
| Column binding | ~minutes (with copies) | In-place `:=` | ~10Ã— |
| Prediction (6.46M rows) | ~hours (if `randomForest`) | ~minutes (chunked, or `ranger`) | ~10â€“60Ã— |
| **End-to-end** | **86+ hours** | **~10â€“30 minutes** | **~200Ã—** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest), spdep (for nb object)
# Preserves: trained RF model object, original numerical estimand
# =============================================================================

library(data.table)

# -------------------------------------------------------------------------
# STEP 1: Build vectorized neighbor edge-list (replaces build_neighbor_lookup)
# -------------------------------------------------------------------------
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt: data.table with columns 'id', 'year', and a '.row_idx' column
  # id_order:     vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
  #

  # Returns: a data.table with columns (row_i, neighbor_row_j)
  #          representing "row_i's neighbor is at neighbor_row_j" for all cell-years.

  message("Building neighbor edge-list...")

  n_cells <- length(id_order)

  # --- 1a. Build cell-level edge list (cell index â†’ neighbor cell indices) ---
  # Flatten the nb list into an edge-list of cell indices within id_order
  from_cell <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_cell   <- unlist(rook_neighbors_unique)

  # Remove 0-neighbor entries (spdep uses integer(0) for no neighbors, which

  # unlist drops, but guard against 0-coded entries)
  valid <- to_cell > 0L & to_cell <= n_cells
  cell_edges <- data.table(
    from_id = id_order[from_cell[valid]],
    to_id   = id_order[to_cell[valid]]
  )

  # --- 1b. Map (id, year) â†’ row index ---
  id_year_map <- cell_data_dt[, .(id, year, .row_idx)]

  # --- 1c. Expand cell edges across all years via join ---
  # For each year, every cell-edge becomes a row-edge.
  # This is the key vectorized step that replaces 6.46M lapply iterations.

  years <- unique(cell_data_dt$year)

  # Cross-join cell_edges Ã— years, then map to row indices
  # Memory-efficient: do in chunks by year (28 years â†’ small loop, each ~1.37M edges)

  edge_chunks <- vector("list", length(years))

  setkey(id_year_map, id, year)

  for (yi in seq_along(years)) {
    yr <- years[yi]

    # Map "from" side
    from_map <- id_year_map[year == yr, .(from_id = id, row_i = .row_idx)]
    setkey(from_map, from_id)

    # Map "to" side
    to_map <- id_year_map[year == yr, .(to_id = id, neighbor_row_j = .row_idx)]
    setkey(to_map, to_id)

    chunk <- cell_edges[from_map, on = "from_id", nomatch = 0L, allow.cartesian = TRUE]
    chunk <- chunk[to_map, on = "to_id", nomatch = 0L, allow.cartesian = TRUE]

    edge_chunks[[yi]] <- chunk[, .(row_i, neighbor_row_j)]
  }

  edge_dt <- rbindlist(edge_chunks)
  setkey(edge_dt, row_i)

  message(sprintf("  Edge-list built: %s edges across %s cell-years.",
                  formatC(nrow(edge_dt), big.mark = ","),
                  formatC(nrow(cell_data_dt), big.mark = ",")))

  return(edge_dt)
}


# -------------------------------------------------------------------------
# STEP 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats)
# -------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data_dt, edge_dt, neighbor_source_vars) {
  # cell_data_dt: data.table with '.row_idx' and all source variable columns
  # edge_dt:      data.table with (row_i, neighbor_row_j), keyed on row_i
  # neighbor_source_vars: character vector of variable names
  #
  # Adds columns <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
  # to cell_data_dt BY REFERENCE. Returns cell_data_dt invisibly.

  message("Computing neighbor features for ", length(neighbor_source_vars), " variables...")

  n_rows <- nrow(cell_data_dt)

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing '%s'...", var_name))

    # Attach the neighbor's value to each edge
    vals <- cell_data_dt[[var_name]]

    # Vectorized: get the neighbor value for every edge
    edge_vals <- data.table(
      row_i = edge_dt$row_i,
      val   = vals[edge_dt$neighbor_row_j]
    )

    # Drop edges where neighbor value is NA
    edge_vals <- edge_vals[!is.na(val)]

    # Grouped aggregation â€” fully vectorized in C inside data.table
    stats <- edge_vals[, .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), by = row_i]

    # Allocate result columns with NA, then fill
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    set(cell_data_dt, j = max_col,  value = NA_real_)
    set(cell_data_dt, j = min_col,  value = NA_real_)
    set(cell_data_dt, j = mean_col, value = NA_real_)

    # Fill in-place by reference using row indices
    set(cell_data_dt, i = stats$row_i, j = max_col,  value = stats$nmax)
    set(cell_data_dt, i = stats$row_i, j = min_col,  value = stats$nmin)
    set(cell_data_dt, i = stats$row_i, j = mean_col, value = stats$nmean)
  }

  message("  Neighbor features complete.")
  invisible(cell_data_dt)
}


# -------------------------------------------------------------------------
# STEP 3: Chunked Random Forest prediction (memory-safe)
# -------------------------------------------------------------------------
predict_rf_chunked <- function(model, newdata_dt, predictor_cols,
                               chunk_size = 500000L) {
  # model:          trained RF model (ranger or randomForest object)
  # newdata_dt:     data.table of prediction data
  # predictor_cols: character vector of the ~110 predictor column names
  # chunk_size:     rows per prediction chunk
  #
  # Returns: numeric vector of predictions, same length as nrow(newdata_dt)

  n <- nrow(newdata_dt)
  preds <- numeric(n)
  n_chunks <- ceiling(n / chunk_size)

  is_ranger <- inherits(model, "ranger")

  message(sprintf("Predicting %s rows in %d chunks of up to %s...",
                  formatC(n, big.mark = ","), n_chunks,
                  formatC(chunk_size, big.mark = ",")))

  for (ci in seq_len(n_chunks)) {
    i_start <- (ci - 1L) * chunk_size + 1L
    i_end   <- min(ci * chunk_size, n)
    idx     <- i_start:i_end

    chunk_df <- as.data.frame(newdata_dt[idx, ..predictor_cols])

    if (is_ranger) {
      chunk_pred <- predict(model, data = chunk_df)$predictions
    } else {
      # randomForest package
      chunk_pred <- predict(model, newdata = chunk_df)
    }

    preds[idx] <- chunk_pred

    if (ci %% 5 == 0 || ci == n_chunks) {
      message(sprintf("  Chunk %d/%d done (%s rows).",
                      ci, n_chunks, formatC(i_end, big.mark = ",")))
    }

    # Free chunk memory
    rm(chunk_df, chunk_pred)
    if (ci %% 10 == 0) gc(verbose = FALSE)
  }

  return(preds)
}


# -------------------------------------------------------------------------
# STEP 4: Conversion helper â€” randomForest â†’ ranger (optional, big speedup)
# -------------------------------------------------------------------------
convert_rf_to_ranger_if_possible <- function(model, train_data, train_y,
                                             predictor_cols) {
  # If the model is a randomForest object and you have the training data,

  # retrain an equivalent ranger model for much faster prediction.
  # NOTE: The problem states "must not be retrained", so this function
  # is provided ONLY as a reference. Use predict_rf_chunked with the
  # original model object to satisfy the constraint.
  #
  # If you determine that using the same hyperparameters and same data
  # constitutes "reproducing" rather than "retraining", this is how:

  if (!inherits(model, "randomForest")) {
    message("Model is already ranger (or other); no conversion needed.")
    return(model)
  }

  message("NOTE: Converting randomForest â†’ ranger for prediction speed.")
  message("      Same ntree, mtry, nodesize. Same training data.")

  library(ranger)
  ranger_model <- ranger(
    x               = train_data[, predictor_cols],
    y               = train_y,
    num.trees       = model$ntree,
    mtry            = model$mtry,
    min.node.size   = if (!is.null(model$nodesize)) model$nodesize else 5,
    num.threads     = parallel::detectCores() - 1L,
    seed            = 42
  )
  return(ranger_model)
}


# =========================================================================
# MAIN PIPELINE â€” drop-in replacement
# =========================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, predictor_cols,
                                   chunk_size = 500000L) {
  # cell_data:              data.frame/data.table with id, year, and all features
  # id_order:               vector of cell IDs matching rook_neighbors_unique
  # rook_neighbors_unique:  spdep nb object (list of integer neighbor indices)
  # rf_model:               trained Random Forest model (ranger or randomForest)
  # predictor_cols:         character vector of ~110 predictor column names
  # chunk_size:             prediction chunk size (tune to available RAM)

  t0 <- Sys.time()

  # --- Convert to data.table and add row index ---
  cell_data_dt <- as.data.table(cell_data)
  cell_data_dt[, .row_idx := .I]

  # --- Step 1: Build vectorized edge-list ---
  edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)

  # --- Step 2: Compute all neighbor features ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  compute_all_neighbor_features(cell_data_dt, edge_dt, neighbor_source_vars)

  # Free edge-list memory
  rm(edge_dt)
  gc(verbose = FALSE)

  # --- Step 3: Predict ---
  cell_data_dt[, predicted_gdp := predict_rf_chunked(
    model          = rf_model,
    newdata_dt     = cell_data_dt,
    predictor_cols = predictor_cols,
    chunk_size     = chunk_size
  )]

  # --- Clean up helper column ---
  cell_data_dt[, .row_idx := NULL]

  t1 <- Sys.time()
  message(sprintf("Pipeline complete in %s.", format(t1 - t0)))

  return(cell_data_dt)
}


# =========================================================================
# USAGE EXAMPLE
# =========================================================================
#
# # Load pre-existing objects
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")
#
# # Define the ~110 predictor column names used during training
# predictor_cols <- readRDS("predictor_cols.rds")
# # OR: predictor_cols <- setdiff(names(cell_data), c("id", "year", "gdp", ...))
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data             = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model              = rf_model,
#   predictor_cols        = predictor_cols,
#   chunk_size            = 500000L
# )
#
# # result is a data.table with all original columns + neighbor features +
# # predicted_gdp. The predicted values are numerically identical to what
# # the original pipeline would produce (same model, same features, same
# # computations â€” just vectorized).
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Concern | Guarantee |
|---|---|
| **Same trained model** | The RF model object is loaded from disk and passed directly to `predict()`. No retraining occurs. |
| **Same numerical estimand** | Neighbor max/min/mean are computed with the same `max()`, `min()`, `mean()` functions on the same neighbor sets. The `data.table` grouped aggregation produces **identical** floating-point results. |
| **Same neighbor definitions** | The edge-list is constructed from the same `rook_neighbors_unique` nb object via the same index-mapping logic â€” just vectorized instead of row-by-row. |
| **Same predictor columns** | `predictor_cols` is explicitly passed and used in the same order. |

---

## 5. KEY PERFORMANCE WINS SUMMARIZED

1. **`build_neighbor_lookup` â†’ `build_neighbor_edgelist`**: Eliminates 6.46M `paste()` + named-vector lookups. Replaced by 28 keyed `data.table` joins (~seconds each).

2. **`compute_neighbor_stats` â†’ grouped `data.table` aggregation**: Eliminates 6.46M Ã— 5 `lapply` iterations and 5 `do.call(rbind, ...)` calls. Replaced by 5 vectorized group-by operations (~seconds each).

3. **Column addition**: `set()` by reference â€” zero copies of the 6.46M-row table.

4. **Prediction**: Chunked to control peak memory; compatible with both `ranger` (fast C++ backend) and `randomForest` (slower but functional). If the model is `randomForest`-class, the chunking at least prevents out-of-memory failures and allows progress tracking.

**Expected total runtime: ~10â€“30 minutes** (down from 86+ hours) on a 16 GB laptop.
END_RESPONSE>>>
