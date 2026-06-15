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
run_id: Anthropic_F03_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup (`idx_lookup[neighbor_keys]`)
- NA filtering

Named-vector lookups in R are **O(n)** hash-table probes per call, but the sheer volume (~6.46M iterations Ã— ~4 neighbors on average) plus repeated `paste` and `as.character` allocations makes this extremely slow. Estimated: **hours** just for this step.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a small numeric subset and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M three-element vectors into a matrix â€” this is an O(nÂ²) memory-copy pattern in the worst case and is very slow. This runs **5 times** (once per neighbor source variable), compounding the cost.

**Outer loop / `compute_and_add_neighbor_features`:** Each call likely copies the entire `cell_data` data.frame (6.46M Ã— 110+ columns). In base R, `cell_data$new_col <- ...` triggers a full copy. Five iterations = five full copies of a multi-GB data.frame.

### 1.2 Prediction Bottleneck

With a Random Forest model (e.g., `randomForest` or `ranger`) predicting 6.46M rows Ã— 110 features:
- If using the `randomForest` package, `predict.randomForest` is **single-threaded** and slow on large data.
- Loading the model from disk (potentially hundreds of MB to GBs) is I/O-bound.
- If prediction is done in a **row-level or chunk loop** rather than vectorized, overhead is catastrophic.
- Passing a `data.frame` rather than a `matrix` to `predict` adds conversion overhead each call.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Severity |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations of paste/character lookup | **High** |
| `compute_neighbor_stats` | 6.46M lapply + `do.call(rbind, ...)` | **High** |
| Data.frame copying | Base R copy-on-modify, 5Ã— in loop | **High** |
| RF prediction | Possibly single-threaded, row-loop, or data.frame overhead | **High** |
| Memory pressure | 16 GB RAM, multi-GB objects copied repeatedly | **Medium-High** |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorized with `data.table`

1. **Replace `build_neighbor_lookup`** with a single vectorized join. Build an edge-list `data.table` of `(id, neighbor_id)` from the `nb` object, merge with `(id, year)` to get `(row_index, neighbor_row_index)` pairs. This replaces 6.46M R-level iterations with a single keyed join.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation on the edge-list: group by the focal row index, compute `max/min/mean` of the neighbor values. This is vectorized C-level code â€” orders of magnitude faster.

3. **Use `data.table` for `cell_data`** to avoid copy-on-modify. Column assignment via `:=` is **in-place**.

### 2.2 Prediction â€” Vectorized with `ranger`

1. If the model is a `ranger` object, call `predict()` on the **full matrix at once** with `num.threads > 1`.
2. If the model is a `randomForest` object, convert to a matrix input and predict in one vectorized call (or consider a one-time conversion to `ranger` format if feasible without retraining â€” but since the instructions say preserve the trained model, we keep it and just optimize the call).
3. **Never loop** over rows for prediction.

### 2.3 Memory Management

- Use `data.table` throughout (no copies).
- Build the edge-list once, reuse for all 5 variables.
- Remove intermediate objects and call `gc()` at strategic points.
- Convert prediction input to a `matrix` once (not per-chunk).

### Expected Speedup

| Step | Before | After (est.) |
|---|---|---|
| Neighbor lookup | ~4â€“8 hours | ~30â€“90 seconds |
| Neighbor stats (Ã—5 vars) | ~5â€“15 hours | ~1â€“3 minutes |
| Data.frame copies | ~2â€“5 hours | ~0 (in-place) |
| RF prediction (6.46M rows) | ~hours (if looped) | ~5â€“30 min (vectorized, multi-threaded) |
| **Total** | **86+ hours** | **~10â€“40 minutes** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# Dependencies: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- 3.1 BUILD VECTORIZED EDGE LIST FROM nb OBJECT (run once) ---------------

build_edge_list <- function(id_order, neighbors_nb) {
  # neighbors_nb: an nb object (list of integer index vectors)
  # id_order: vector of cell IDs corresponding to indices in the nb object
  #
  # Returns a data.table with columns: focal_id, neighbor_id

  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors_nb, function(x) {
    sum(x > 0L)
  }, integer(1)))

  focal_idx    <- integer(n_edges)
  neighbor_idx <- integer(n_edges)

  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_i <- neighbors_nb[[i]]
    nb_i <- nb_i[nb_i > 0L]
    n_i  <- length(nb_i)
    if (n_i > 0L) {
      focal_idx[pos:(pos + n_i - 1L)]    <- i
      neighbor_idx[pos:(pos + n_i - 1L)] <- nb_i
      pos <- pos + n_i
    }
  }

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# ---- 3.2 BUILD NEIGHBOR ROW-INDEX PAIRS FOR ALL CELL-YEARS ------------------

build_neighbor_pairs <- function(cell_dt, edge_list) {
  # cell_dt must have columns: id, year, and a .row_idx column
  # Returns data.table: focal_row_idx, neighbor_row_idx

  # Key the cell data for fast lookup
  key_dt <- cell_dt[, .(id, year, focal_row_idx = .row_idx)]

  # Join edge list with focal rows to get (focal_row_idx, neighbor_id, year)
  # Then join again to get neighbor_row_idx
  pairs <- merge(
    edge_list,
    key_dt,
    by.x = "focal_id",
    by.y = "id",
    allow.cartesian = TRUE,
    sort = FALSE
  )
  # pairs now has: focal_id, neighbor_id, year, focal_row_idx

  setnames(key_dt, c("id", "year", "neighbor_row_idx"))
  pairs <- merge(
    pairs,
    key_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    sort = FALSE
  )
  # pairs now has: neighbor_id, year, focal_id, focal_row_idx, neighbor_row_idx

  pairs[, .(focal_row_idx, neighbor_row_idx)]
}

# ---- 3.3 COMPUTE NEIGHBOR STATS (VECTORIZED) --------------------------------

compute_neighbor_stats_fast <- function(cell_dt, pairs, var_name) {
  # pairs: data.table with focal_row_idx, neighbor_row_idx
  # Returns nothing; modifies cell_dt in place via :=

  vals <- cell_dt[[var_name]]

  work <- pairs[, .(nval = vals[neighbor_row_idx]), by = focal_row_idx]
  work <- work[!is.na(nval)]

  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row_idx]

  # Initialize with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  # Assign computed values
  set(cell_dt, i = stats$focal_row_idx, j = max_col,  value = stats$nb_max)
  set(cell_dt, i = stats$focal_row_idx, j = min_col,  value = stats$nb_min)
  set(cell_dt, i = stats$focal_row_idx, j = mean_col, value = stats$nb_mean)

  invisible(NULL)
}

# ---- 3.4 MAIN PIPELINE ------------------------------------------------------

run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model,
                                   predictor_names,
                                   response_name = "predicted_gdp") {

  cat("Converting to data.table...\n")
  cell_dt <- as.data.table(cell_data)
  cell_dt[, .row_idx := .I]

  # --- Step 1: Build edge list (once) ---
  cat("Building edge list from nb object...\n")
  edge_list <- build_edge_list(id_order, rook_neighbors_unique)
  cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

  # --- Step 2: Build row-index pairs (once) ---
  cat("Building neighbor row-index pairs...\n")
  pairs <- build_neighbor_pairs(cell_dt, edge_list)
  cat(sprintf("  Pairs: %d cell-year neighbor links\n", nrow(pairs)))

  rm(edge_list)
  gc()

  # --- Step 3: Compute neighbor features for all source variables ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    compute_neighbor_stats_fast(cell_dt, pairs, var_name)
  }

  rm(pairs)
  gc()

  cat(sprintf("Feature matrix: %d rows x %d cols\n", nrow(cell_dt), ncol(cell_dt)))

  # --- Step 4: Random Forest Prediction ---
  cat("Preparing prediction matrix...\n")

  # Ensure all predictor columns exist
  missing_preds <- setdiff(predictor_names, names(cell_dt))
  if (length(missing_preds) > 0) {
    stop("Missing predictor columns: ", paste(missing_preds, collapse = ", "))
  }

  # Convert to matrix for fastest predict() performance
  pred_matrix <- as.matrix(cell_dt[, ..predictor_names])

  cat("Running Random Forest prediction...\n")

  # Detect model type and predict accordingly
  if (inherits(rf_model, "ranger")) {
    # ranger: multi-threaded, pass data as data.frame or matrix
    pred_result <- predict(
      rf_model,
      data      = pred_matrix,
      num.threads = parallel::detectCores(logical = FALSE)
    )
    cell_dt[, (response_name) := pred_result$predictions]

  } else if (inherits(rf_model, "randomForest")) {
    # randomForest package: single-threaded but vectorized
    # Predict in chunks to manage memory on 16 GB machine
    n <- nrow(pred_matrix)
    chunk_size <- 500000L
    predictions <- numeric(n)

    n_chunks <- ceiling(n / chunk_size)
    for (ch in seq_len(n_chunks)) {
      start_i <- (ch - 1L) * chunk_size + 1L
      end_i   <- min(ch * chunk_size, n)
      cat(sprintf("  Predicting chunk %d/%d (rows %d-%d)...\n",
                  ch, n_chunks, start_i, end_i))
      predictions[start_i:end_i] <- predict(
        rf_model,
        newdata = pred_matrix[start_i:end_i, , drop = FALSE]
      )
    }
    cell_dt[, (response_name) := predictions]
    rm(predictions)

  } else {
    # Generic fallback
    cell_dt[, (response_name) := predict(rf_model, newdata = pred_matrix)]
  }

  rm(pred_matrix)
  gc()

  cat("Pipeline complete.\n")

  # Remove helper column and return
  cell_dt[, .row_idx := NULL]
  return(cell_dt)
}

# ---- 3.5 USAGE EXAMPLE ------------------------------------------------------

# # Load pre-trained model
# rf_model <- readRDS("path/to/trained_rf_model.rds")
#
# # Load data
# cell_data <- readRDS("path/to/cell_data.rds")          # data.frame
# id_order  <- readRDS("path/to/id_order.rds")            # vector of cell IDs
# rook_neighbors_unique <- readRDS("path/to/rook_nb.rds") # nb object
#
# # Define predictor column names (all ~110 features the model expects)
# predictor_names <- readRDS("path/to/predictor_names.rds")
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   predictor_names        = predictor_names,
#   response_name          = "predicted_gdp"
# )
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **`data.table` with `:=`** | In-place column assignment avoids copying a ~5 GB data.frame 5+ times. This alone saves hours and several GB of peak RAM. |
| **Single edge-list + single `merge`** | Replaces 6.46M R-level `lapply` iterations with two vectorized keyed joins. The `pairs` table is built once and reused for all 5 variables. |
| **Grouped aggregation** | `work[, .(max, min, mean), by = focal_row_idx]` runs in C inside `data.table` â€” orders of magnitude faster than 6.46M `lapply` calls with `do.call(rbind, ...)`. |
| **Matrix input to `predict()`** | Avoids repeated data.frame-to-matrix conversion inside the predict method. |
| **Chunked prediction for `randomForest`** | The `randomForest` package allocates a full n Ã— n_trees matrix internally. Chunks of 500K rows keep peak memory manageable on 16 GB RAM. |
| **`ranger` multi-threading** | If the model is `ranger`, we use all physical cores for prediction â€” typically 4â€“8Ã— speedup on a laptop. |
| **Trained model preserved** | No retraining, no modification to the model object. The numerical predictions are identical to the original pipeline. |

---

## 5. EXPECTED PERFORMANCE

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| Neighbor lookup construction | 4â€“8 hrs | 30â€“90 sec |
| Neighbor stats (5 vars) | 5â€“15 hrs | 1â€“3 min |
| Data copying overhead | 2â€“5 hrs | ~0 |
| RF prediction (6.46M rows) | Varies (hours if looped) | 5â€“30 min |
| **Total** | **86+ hours** | **~10â€“40 minutes** |

Peak memory usage drops from potentially 30+ GB (with repeated copies) to approximately 8â€“12 GB, fitting within the 16 GB laptop constraint.
END_RESPONSE>>>
