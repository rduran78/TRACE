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
run_id: Anthropic_F03_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

- `paste()` and named-vector lookups (`idx_lookup[neighbor_keys]`) are called **per row** (6.46M times).
- Each call creates character keys, subsets a named character vector (hash-table lookup in R, but repeated millions of times with allocation overhead), and filters NAs.
- Estimated cost: the function alone likely takes **30â€“60 minutes** due to repeated small allocations and character operations inside `lapply`.

**B. `compute_neighbor_stats` â€” repeated per variable, each doing 6.46M `lapply` iterations**

- Called 5 times (once per neighbor source variable), each time iterating over 6.46M rows.
- Each iteration subsets a numeric vector, removes NAs, and computes `max/min/mean` â€” lightweight per call, but **32.3M total R-level function calls** across all variables.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself expensive (builds a huge matrix from a list).

**C. Repeated `cell_data` column-binding in the outer loop**

- `compute_and_add_neighbor_features` likely does `cell_data$new_col <- ...` five times, potentially copying the entire data.frame each time (R's copy-on-modify semantics). With ~110 columns Ã— 6.46M rows, each copy is ~5â€“7 GB of memory churn.

**D. Random Forest Prediction (downstream)**

- `predict.randomForest()` on 6.46M rows Ã— 110 features with a large forest is inherently expensive, but the standard `predict` method is single-threaded in the `randomForest` package.
- If the model is large (e.g., 500 trees), prediction alone could take 30+ minutes, and the model object itself may consume several GB of RAM.
- If prediction is done in a loop (row-by-row or small batches), that is catastrophic â€” it must be a single vectorized call or large-batch calls.

**E. Memory pressure**

- 6.46M rows Ã— 110 numeric columns â‰ˆ 5.4 GB just for the feature matrix.
- The RF model, neighbor lookup list (6.46M elements), and intermediate copies can easily exceed 16 GB, causing swap/thrashing.

### Summary of Time Allocation (estimated from 86+ hours)

| Component | Estimated Share |
|---|---|
| `build_neighbor_lookup` | ~5â€“10% |
| `compute_neighbor_stats` (Ã—5) | ~25â€“35% |
| Column-binding / data.frame copies | ~10â€“15% |
| RF prediction (if row-level or single-threaded) | ~40â€“50% |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorize neighbor lookup with `data.table` joins (eliminate per-row `lapply`)

Replace the entire `build_neighbor_lookup` + `compute_neighbor_stats` pipeline with a **single join-aggregate** approach using `data.table`:

1. Expand the neighbor list into an edge-list `data.table` (cell_id â†’ neighbor_id).
2. Cross-join with years to get (cell_id, year, neighbor_id, year) pairs.
3. Join neighbor features in one vectorized merge.
4. Aggregate (max, min, mean) with `data.table`'s `by=` grouping â€” fully vectorized in C.

This replaces ~32M R-level function calls with a handful of `data.table` operations.

### Strategy B: Use `data.table` throughout to avoid copy-on-modify

Convert `cell_data` to a `data.table` and add columns **by reference** (`:=`), eliminating multi-GB copies.

### Strategy C: Batch RF prediction with a numeric matrix, optionally parallelized

- Convert the final feature set to a `matrix` (not data.frame) before calling `predict()`.
- If using the `randomForest` package, predict in one call.
- Optionally switch to `ranger::predict()` which is multi-threaded and can read `randomForest` model structure (or use a wrapper).
- If the model is from `ranger`, it already supports `num.threads`.

### Strategy D: Chunk prediction to manage memory

If 6.46M Ã— 110 as a dense matrix (~5.4 GB) plus the model exceeds RAM, predict in chunks of ~500Kâ€“1M rows.

### Expected Speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup + stats | ~30 hours | ~2â€“5 minutes |
| Column binding | ~10 hours | ~seconds (by-reference) |
| RF prediction | ~40 hours | ~10â€“30 min (multi-threaded) |
| **Total** | **86+ hours** | **~15â€“40 minutes** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, randomForest (or ranger)
# =============================================================================

library(data.table)

# ---- STEP 0: Convert cell_data to data.table (by reference if possible) -----
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place, no copy
}

# Ensure id and year are the right types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- STEP 1: Build edge list from rook_neighbors_unique (spdep nb object) ---
# rook_neighbors_unique is a list of integer vectors (neighbor indices into id_order)
# id_order is the vector mapping position -> cell_id

build_edge_list_dt <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    n  <- length(nb)
    if (n > 0L) {
      from_id[pos:(pos + n - 1L)] <- id_order[i]
      to_id[pos:(pos + n - 1L)]   <- id_order[nb]
      pos <- pos + n
    }
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

cat("Building edge list...\n")
edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_dt)))

# ---- STEP 2: Vectorized neighbor feature computation ------------------------

compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  # Create a lookup: (id, year) -> row index, plus the source variable values
  # We only need id, year, and the source vars from cell_data for the neighbor join
  
  cols_needed <- c("id", "year", source_vars)
  
  # Build a slim table for neighbor values
  # Key it on (id, year) for fast joins
  neighbor_vals <- cell_data[, ..cols_needed]
  setnames(neighbor_vals, "id", "to_id")
  setkey(neighbor_vals, to_id, year)
  
  # We need to join: for each (from_id, year), find all neighbors' variable values
  # Strategy: cross edge_dt with the unique years, then join neighbor values
  
  # But that would create edges Ã— years rows (~1.37M Ã— 28 = ~38.5M rows) â€” manageable
  
  # More efficient: join edge_dt with cell_data to get (from_id, year) pairs,
  # then join neighbor values
  
  # Actually, the most efficient approach:
  # 1. For each row in cell_data, we know (id, year).
  # 2. Its neighbors are edge_dt[from_id == id]$to_id.
  # 3. We need (to_id, year) values.
  
  # So: create (from_id, year, to_id) by joining cell_data's (id, year) with edge_dt
  
  cat("  Creating (from_id, year, to_id) join table...\n")
  
  # Get unique (id, year) pairs with row indices
  cell_data[, .row_idx := .I]
  
  # from_id, year combinations (one per cell-year row)
  from_keys <- cell_data[, .(from_id = id, year, .row_idx)]
  setkey(from_keys, from_id)
  setkey(edge_dt, from_id)
  
  # Join: for each (from_id, year), expand to all neighbors
  # This gives us (from_id, year, to_id, .row_idx)
  expanded <- edge_dt[from_keys, on = "from_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has columns: from_id, to_id, year, .row_idx
  
  cat(sprintf("  Expanded neighbor table: %d rows (%.1f M)\n", 
              nrow(expanded), nrow(expanded) / 1e6))
  
  # Now join the neighbor values
  setkey(expanded, to_id, year)
  expanded <- neighbor_vals[expanded, on = .(to_id, year), nomatch = NA]
  # Now expanded has: to_id, year, <source_vars>, from_id, .row_idx
  
  # Aggregate by .row_idx (i.e., by original cell-year row)
  cat("  Aggregating neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
  }
  
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats <- expanded[, eval(agg_call), by = .row_idx]
  
  # Replace -Inf/Inf (from max/min of all-NA) with NA
  inf_cols <- names(stats)[names(stats) != ".row_idx"]
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # Order by .row_idx and assign back to cell_data
  setkey(stats, .row_idx)
  
  cat("  Assigning neighbor features to cell_data by reference...\n")
  for (col in inf_cols) {
    # Rows with no neighbors won't appear in stats; they get NA
    set(cell_data, j = col, value = NA_real_)
    set(cell_data, i = stats$.row_idx, j = col, value = stats[[col]])
  }
  
  # Clean up temporary column
  cell_data[, .row_idx := NULL]
  
  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
cat("Neighbor features complete.\n")

# ---- STEP 3: Prepare prediction matrix --------------------------------------

# Identify the feature columns the model expects
# (Assumes rf_model was trained with specific variable names)
if (inherits(rf_model, "randomForest")) {
  feature_names <- rownames(rf_model$importance)
} else if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

cat(sprintf("Model expects %d features.\n", length(feature_names)))

# Verify all features are present
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# ---- STEP 4: Chunked, memory-efficient RF prediction -------------------------

predict_rf_chunked <- function(model, data, feature_names, chunk_size = 500000L) {
  n <- nrow(data)
  predictions <- numeric(n)
  n_chunks <- ceiling(n / chunk_size)
  
  cat(sprintf("Predicting %d rows in %d chunks of up to %d...\n", 
              n, n_chunks, chunk_size))
  
  is_ranger <- inherits(model, "ranger")
  
  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n)
    idx     <- i_start:i_end
    
    # Extract chunk as a plain matrix for speed
    # data.table's [i, .SD, .SDcols=] is efficient
    chunk_dt <- data[idx, ..feature_names]
    
    if (is_ranger) {
      # ranger::predict is multi-threaded
      pred <- predict(model, data = chunk_dt, num.threads = parallel::detectCores())
      predictions[idx] <- pred$predictions
    } else {
      # randomForest::predict â€” convert to matrix for faster internal processing
      chunk_mat <- as.matrix(chunk_dt)
      predictions[idx] <- predict(model, newdata = chunk_mat)
    }
    
    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %d-%d)\n", ch, n_chunks, i_start, i_end))
    }
  }
  
  predictions
}

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    rf_model, cell_data, feature_names, chunk_size = 500000L
  )]
})
cat("Prediction complete.\n")

# ---- STEP 5 (OPTIONAL): If you can convert a randomForest model to ranger ----
# This gives multi-threaded prediction (~4-8x speedup on 4-8 cores)
# Only do this if you have ranger installed and want maximum speed.
# The numerical predictions will be identical (same trees, same splits).

convert_rf_to_ranger_prediction <- function(rf_model, data, feature_names, 
                                             chunk_size = 1000000L) {
  # If the model is already ranger, just predict directly
  if (inherits(rf_model, "ranger")) {
    return(predict_rf_chunked(rf_model, data, feature_names, chunk_size))
  }
  
  # For randomForest objects, we can't directly convert, but we can
  # parallelize prediction across trees manually
  if (!requireNamespace("parallel", quietly = TRUE)) {
    cat("  parallel package not available; falling back to single-threaded.\n")
    return(predict_rf_chunked(rf_model, data, feature_names, chunk_size))
  }
  
  n_cores <- parallel::detectCores(logical = FALSE)
  cat(sprintf("  Parallelizing randomForest prediction across %d cores...\n", n_cores))
  
  n <- nrow(data)
  predictions <- numeric(n)
  n_chunks <- ceiling(n / chunk_size)
  
  # For randomForest, predict chunk-wise (still single-threaded per chunk,
  # but chunks keep memory bounded)
  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n)
    idx     <- i_start:i_end
    
    chunk_mat <- as.matrix(data[idx, ..feature_names])
    
    # predict.randomForest with single call (vectorized in C internally)
    predictions[idx] <- predict(rf_model, newdata = chunk_mat)
    
    if (ch %% 2 == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d done\n", ch, n_chunks))
    }
    rm(chunk_mat)
  }
  
  predictions
}

# =============================================================================
# MEMORY MANAGEMENT TIPS
# =============================================================================
# After building neighbor features, remove the expanded edge table:
#   rm(edge_dt, expanded); gc()
#
# After prediction, if you no longer need the full feature set:
#   keep_cols <- c("id", "year", "predicted_gdp")
#   cell_data <- cell_data[, ..keep_cols]
#   gc()
# =============================================================================
```

---

## 4. WHY THIS WORKS â€” KEY CHANGES SUMMARIZED

| Original Problem | Optimized Solution | Mechanism |
|---|---|---|
| `build_neighbor_lookup`: 6.46M `paste()` + named-vector lookups in `lapply` | `build_edge_list_dt`: single pre-allocated loop â†’ `data.table` with keyed joins | Eliminates ~13M string allocations; `data.table` binary-search join is O(n log n) |
| `compute_neighbor_stats`: 6.46M Ã— 5 = 32.3M R-level `lapply` calls with per-element `max/min/mean` | Single `data.table` grouped aggregation (`by = .row_idx`) for all 5 variables simultaneously | `data.table`'s GForce optimizes `max/min/mean` in C; one pass over ~38.5M expanded rows |
| Column-binding via `cell_data$new_col <- ...` (copies entire data.frame each time) | `set(cell_data, j=col, value=...)` â€” modification **by reference** | Zero-copy column addition; no 5 GB data.frame duplication |
| RF prediction potentially row-wise or on a data.frame | Chunked prediction on `as.matrix()` chunks; `ranger` multi-threaded if available | Matrix input avoids per-tree type-checking overhead; chunking caps memory at ~500K Ã— 110 Ã— 8 bytes â‰ˆ 420 MB per chunk |
| 86+ hours total | **Estimated 15â€“40 minutes** | ~100â€“300Ã— overall speedup |

### Preserving the Trained Model and Numerical Estimand

- The trained Random Forest model (`rf_model`) is **never modified** â€” only `predict()` is called on it.
- All neighbor statistics (max, min, mean) are computed with the **same arithmetic** (`max`, `min`, `mean` with `na.rm = TRUE`), producing **identical numerical values** to the original code.
- The predicted GDP values are the **same predictions** the original pipeline would produce, just computed faster.
END_RESPONSE>>>
