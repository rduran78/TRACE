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
run_id: Anthropic_F03_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all inside an interpreted R loop. |
| **Repeated `paste` key construction** | `paste(neighbor_cell_ids, data$year[i], sep="_")` is called ~6.46M times, each producing a small character vector. This is extremely GC-heavy. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | Named vector indexing in R uses linear hashing per call; with millions of calls the overhead is enormous. |
| **Output is a list of 6.46M integer vectors** | This alone consumes several GB and causes heavy memory fragmentation. |

**`compute_neighbor_stats`** is the second bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small integer vector, subsets `vals`, removes NAs, computes 3 summary stats. Pure R overhead per element is ~microseconds Ã— 6.46M = hours. |
| **`do.call(rbind, result)` on 6.46M single-row results** | This is a known R anti-pattern; it copies and re-allocates repeatedly. |
| **Called 5 times (once per variable)** | Total: ~32.3M interpreted iterations just for neighbor stats. |

**Outer loop** calls `compute_and_add_neighbor_features` 5 times, each presumably rebuilding or re-traversing the 6.46M-element lookup. No vectorization or `data.table` grouping is used.

### B. Random Forest Inference Bottleneck

With ~6.46M rows Ã— 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object will:

- Allocate a prediction matrix of ~6.46M Ã— 110 doubles (~5.3 GB).
- Traverse every tree for every row (CPU-bound but unavoidable).
- If using `randomForest::predict`, the implementation is single-threaded and copies the data frame internally.
- If the model is loaded from disk each time, deserialization of a large RF object can take minutes.

### C. Memory Pressure

On a 16 GB laptop, the data (~6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.3 GB) plus the neighbor lookup list (~2â€“4 GB) plus the RF model (~1â€“4 GB) plus intermediate copies can easily exceed RAM, causing swapping and catastrophic slowdown.

### Summary: Where the 86+ Hours Go

| Phase | Estimated Share |
|---|---|
| `build_neighbor_lookup` (R-loop, paste, named lookup) | ~30â€“40% |
| `compute_neighbor_stats` Ã— 5 vars (R-loop, rbind) | ~30â€“40% |
| RF `predict()` (single-thread, data copy) | ~15â€“25% |
| Memory pressure / GC / swapping | ~10â€“20% |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Eliminate interpreted R loops; vectorize everything via `data.table` joins and columnar operations.

| Original Approach | Optimized Approach |
|---|---|
| Build a 6.46M-element list via `lapply` + `paste` keys | Build a flat `data.table` edge list; merge via keyed join |
| Compute neighbor stats via `lapply` + `do.call(rbind,â€¦)` | Compute neighbor stats via `data.table` grouped aggregation (`[, .(max, min, mean), by=]`) |
| Named-vector lookup for row indices | Integer-keyed `data.table` binary-search join |
| `randomForest::predict` (single-threaded) | `ranger::predict` or chunked prediction with `num.threads` |
| Load model from disk each run | Load once, keep in memory; use `qs` or `fst` for fast deserialization |
| Full data frame copy for `predict()` | Predict in-place on a `data.table` matrix; chunk if memory-constrained |

**Expected speedup: from 86+ hours to ~10â€“30 minutes.**

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- 0. FAST MODEL LOADING ------------------------------------------------
# Load the trained RF model once. If stored as .rds, consider converting to
# qs format for faster deserialization:
#   qs::qsave(rf_model, "rf_model.qs")
#   rf_model <- qs::qread("rf_model.qs")
# Otherwise:
rf_model <- readRDS("rf_model.rds")


# ---- 1. CONVERT DATA TO data.table ----------------------------------------
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2,
# plus all other predictor columns.

setDT(cell_data)

# Create a unique integer row key for fast joining
cell_data[, row_idx := .I]


# ---- 2. BUILD FLAT NEIGHBOR EDGE LIST (VECTORIZED) ------------------------
# rook_neighbors_unique: an nb object (list of integer vectors).
# id_order: vector of cell IDs in the order matching the nb object.
#
# This replaces build_neighbor_lookup entirely.

build_neighbor_edges_dt <- function(id_order, neighbors) {
  # Expand the nb list into a two-column edge table: (focal_id, neighbor_id)
  # Each element neighbors[[i]] contains integer indices into id_order.
  
  n <- length(neighbors)
  
  # Count edges to pre-allocate
  lens <- vapply(neighbors, length, integer(1))
  total_edges <- sum(lens)
  
  # Pre-allocate vectors
  focal_idx    <- rep(seq_len(n), times = lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  # Remove zero-neighbor sentinels (spdep uses 0 for no neighbors)
  valid <- neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  # Map to actual cell IDs
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

edge_dt <- build_neighbor_edges_dt(id_order, rook_neighbors_unique)

cat(sprintf("Edge table: %d directed neighbor relationships\n", nrow(edge_dt)))


# ---- 3. COMPUTE ALL NEIGHBOR FEATURES (VECTORIZED) ------------------------
# Strategy:
#   1. Join edge_dt with cell_data on (neighbor_id, year) to get neighbor values.
#   2. Group by (focal_id, year) and compute max/min/mean.
#   3. Join results back to cell_data.
#
# This replaces both compute_neighbor_stats and the outer for-loop.

compute_all_neighbor_features_dt <- function(cell_data, edge_dt, var_names) {
  
  # Minimal lookup table: id, year, and the source variables only
  lookup_cols <- c("id", "year", var_names)
  lookup <- cell_data[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  
  # Key for fast binary-search join
  setkey(lookup, neighbor_id, year)
  setkey(edge_dt, neighbor_id)  # will also use year from lookup
  
  # Expand edges Ã— years: for each (focal_id, year), find neighbor values.
  # We join edge_dt to cell_data to get the year dimension for the focal cell,
  # then join again on (neighbor_id, year) to get neighbor values.
  
  # Step A: Get (focal_id, year) pairs from cell_data
  focal_years <- cell_data[, .(focal_id = id, year)]
  
  # Step B: Merge focal_years with edge_dt to get (focal_id, year, neighbor_id)
  setkey(edge_dt, focal_id)
  expanded <- edge_dt[focal_years, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: focal_id, neighbor_id, year
  
  cat(sprintf("Expanded edge-year table: %d rows\n", nrow(expanded)))
  
  # Step C: Join to get neighbor variable values
  setkey(expanded, neighbor_id, year)
  setkey(lookup, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, focal_id, + var_names columns
  
  # Step D: Grouped aggregation â€” compute max, min, mean for each variable
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in var_names) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <- bquote(max(.(v_sym),  na.rm = TRUE))
    agg_exprs[[paste0("nb_min_", v)]]  <- bquote(min(.(v_sym),  na.rm = TRUE))
    agg_exprs[[paste0("nb_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Convert to a single call for data.table's j
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  cat("Computing grouped neighbor statistics...\n")
  stats <- expanded[, eval(agg_call), by = .(focal_id, year)]
  
  # Replace Inf/-Inf (from max/min on all-NA groups) with NA
  inf_cols <- names(stats)[-(1:2)]
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  # Step E: Join back to cell_data
  setkey(stats, focal_id, year)
  setkey(cell_data, id, year)
  cell_data <- stats[cell_data, on = .(focal_id = id, year)]
  
  # Rename focal_id back to id
  setnames(cell_data, "focal_id", "id")
  
  return(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_dt(cell_data, edge_dt, neighbor_source_vars)

cat(sprintf("Feature preparation complete. %d rows, %d columns.\n",
            nrow(cell_data), ncol(cell_data)))

# Clean up large intermediate objects
rm(edge_dt)
gc()


# ---- 4. RANDOM FOREST PREDICTION (OPTIMIZED) ------------------------------
# Key optimizations:
#   - Use ranger::predict if model is ranger (multi-threaded by default).
#   - If model is randomForest, predict in chunks to control memory.
#   - Prepare a plain matrix of predictors (avoids data.frame overhead).

predict_rf_optimized <- function(cell_data, rf_model, predictor_names,
                                  chunk_size = 500000L) {
  
  n <- nrow(cell_data)
  model_class <- class(rf_model)[1]
  
  cat(sprintf("Predicting %d rows with %s model...\n", n, model_class))
  
  if (model_class == "ranger") {
    # ranger::predict is already multi-threaded and memory-efficient.
    # Provide a data.frame (required by ranger), but only predictor columns.
    pred_data <- as.data.frame(cell_data[, ..predictor_names])
    preds <- predict(rf_model, data = pred_data, num.threads = parallel::detectCores())
    cell_data[, predicted_gdp := preds$predictions]
    rm(pred_data, preds)
    
  } else if (model_class == "randomForest") {
    # randomForest::predict is single-threaded and copies data internally.
    # Chunk to avoid memory blowup.
    cell_data[, predicted_gdp := NA_real_]
    
    n_chunks <- ceiling(n / chunk_size)
    cat(sprintf("Processing in %d chunks of up to %d rows...\n", n_chunks, chunk_size))
    
    for (i in seq_len(n_chunks)) {
      start_row <- (i - 1L) * chunk_size + 1L
      end_row   <- min(i * chunk_size, n)
      idx       <- start_row:end_row
      
      chunk_df <- as.data.frame(cell_data[idx, ..predictor_names])
      preds    <- predict(rf_model, newdata = chunk_df)
      
      set(cell_data, i = idx, j = "predicted_gdp", value = as.numeric(preds))
      
      rm(chunk_df, preds)
      if (i %% 5 == 0) gc()  # periodic GC to reclaim memory
      
      if (i %% max(1, n_chunks %/% 10) == 0) {
        cat(sprintf("  Chunk %d/%d complete\n", i, n_chunks))
      }
    }
    
  } else {
    # Generic fallback â€” try predict() directly
    pred_data <- as.data.frame(cell_data[, ..predictor_names])
    preds <- predict(rf_model, newdata = pred_data)
    cell_data[, predicted_gdp := as.numeric(preds)]
    rm(pred_data, preds)
  }
  
  gc()
  cat("Prediction complete.\n")
  return(cell_data)
}

# Define predictor column names (all columns except id, year, and the target).
# Adjust this to match your actual predictor set:
non_predictor_cols <- c("id", "year", "predicted_gdp", "row_idx")
predictor_names <- setdiff(names(cell_data), non_predictor_cols)

# Ensure predictor_names matches what the model expects
if (inherits(rf_model, "ranger")) {
  expected <- rf_model$forest$independent.variable.names
  predictor_names <- intersect(expected, names(cell_data))
} else if (inherits(rf_model, "randomForest")) {
  expected <- rownames(importance(rf_model))
  predictor_names <- intersect(expected, names(cell_data))
}

cat(sprintf("Using %d predictor variables.\n", length(predictor_names)))

cell_data <- predict_rf_optimized(cell_data, rf_model, predictor_names)


# ---- 5. FINAL OUTPUT ------------------------------------------------------
# Remove helper columns
cell_data[, row_idx := NULL]

cat(sprintf("Done. Output: %d rows, %d columns.\n", nrow(cell_data), ncol(cell_data)))
cat(sprintf("Predicted GDP range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))
```

---

## 4. EXPECTED PERFORMANCE COMPARISON

| Phase | Original (est.) | Optimized (est.) | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~20â€“30 hrs | ~10â€“30 sec (vectorized edge table) | ~3,000Ã— |
| `compute_neighbor_stats` Ã— 5 | ~30â€“40 hrs | ~2â€“8 min (data.table grouped agg) | ~300Ã— |
| RF prediction (6.46M rows) | ~10â€“15 hrs | ~5â€“20 min (multi-thread / chunked) | ~30â€“60Ã— |
| Memory overhead / GC | ~10+ hrs | negligible | â€” |
| **Total** | **~86+ hrs** | **~10â€“30 min** | **~200Ã—** |

### Key Invariants Preserved

- **Trained RF model**: loaded from disk, never retrained; `predict()` only.
- **Numerical estimand**: the same max/min/mean neighbor statistics are computed for the same 5 variables; the same predictor columns are fed to the same model; predicted values are numerically identical.
- **Spatial structure**: the rook-neighbor relationships from `rook_neighbors_unique` are faithfully expanded into the edge table with no alteration.
END_RESPONSE>>>
