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
run_id: Anthropic_F03_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys, then named-vector lookup (`idx_lookup[neighbor_keys]`)
- NA filtering

Named-vector lookups in R are **O(n)** hash probes per call, but the sheer volume (~6.46M iterations Ã— multiple lookups each) makes this extremely slow. The function also creates ~6.46M small integer vectors, which hammers R's memory allocator.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset, removing NAs, and computing three summary statistics. The final `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself a major bottleneck (repeated memory allocation and copying).

**Outer loop over 5 variables:** `compute_and_add_neighbor_features` is called 5 times, each time iterating over all 6.46M rows. If `compute_and_add_neighbor_features` copies `cell_data` (likely, since `cell_data` is a data.frame being modified in a loop), each iteration triggers a full copy of a ~6.46M Ã— 110+ column data.frame (~5â€“8 GB).

### 1.2 Random Forest Inference Bottleneck

Calling `predict()` on a single Random Forest model with ~6.46M rows and ~110 features is:
- **Memory-intensive:** `ranger` or `randomForest` must allocate prediction matrices. With `randomForest`, the default `predict.randomForest` converts input to a matrix internally.
- **Slow in one shot:** If using `randomForest` (not `ranger`), prediction is single-threaded and R-level tree traversal is slow.
- **Object copying risk:** If the prediction input is a `data.frame` that gets coerced to a matrix, that's a ~5.4 GB temporary allocation (6.46M Ã— 110 Ã— 8 bytes).

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops | Hours |
| `compute_neighbor_stats` (Ã—5) | 6.46M R-level iterations + `do.call(rbind, ...)` | Hours per variable |
| Data.frame copy-on-modify | `cell_data` copied each loop iteration | ~5â€“8 GB Ã— 5 copies |
| RF prediction | Possibly single-threaded, full matrix coercion | 30 minâ€“hours |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorize neighbor lookup using `data.table` joins
Replace the row-by-row `lapply` with a single merge/join operation. Pre-build an edge list (cell-year â†’ neighbor-cell-year) and join to get neighbor values, then group-by to compute stats.

### Strategy B: Vectorized neighbor stats via `data.table` grouped aggregation
Instead of iterating per row, use `data.table`'s `[, .(max, min, mean), by = ...]` on the joined edge table.

### Strategy C: Use `data.table` by reference to avoid copies
Replace `data.frame` with `data.table` and use `:=` to add columns in place â€” zero copies.

### Strategy D: Use `ranger` for prediction (or batch `randomForest`)
If the model is `ranger`, use `num.threads`. If `randomForest`, convert input to matrix once and predict in chunks to control memory.

### Estimated speedup: from 86+ hours â†’ ~10â€“30 minutes.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# =============================================================================
# Requirements: data.table, ranger (if model is ranger), randomForest (if model
# is randomForest). The trained RF model object is preserved as-is.
# =============================================================================

library(data.table)

# ---- STEP 0: Convert cell_data to data.table (in place, no copy) -----------

if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ---- STEP 1: Build edge list from spdep::nb object -------------------------
# rook_neighbors_unique is a list of integer vectors (nb object).
# id_order is the vector mapping position -> cell id.
# We build a data.table: (focal_id, neighbor_id)

build_edge_list_dt <- function(id_order, neighbors) {
  # neighbors[[i]] gives the neighbor positions for position i in id_order
  n <- length(neighbors)
  
  # Pre-calculate total edges for pre-allocation
  edge_counts <- vapply(neighbors, function(x) {
    # spdep::nb uses 0L to indicate no neighbors
    sum(x > 0L)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  focal_pos   <- integer(total_edges)
  neighbor_pos <- integer(total_edges)
  
  offset <- 0L
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    nb <- nb[nb > 0L]
    k  <- length(nb)
    if (k > 0L) {
      idx <- offset + seq_len(k)
      focal_pos[idx]    <- i
      neighbor_pos[idx] <- nb
      offset <- offset + k
    }
  }
  
  data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_dt)))

# ---- STEP 2: Vectorized neighbor feature computation -----------------------
# For each (focal_id, year), join to neighbor rows and compute max/min/mean.

compute_all_neighbor_features_dt <- function(cell_data, edge_dt, 
                                              neighbor_source_vars) {
  # Create a row key for fast joining
  # We join: edge_dt (focal_id, neighbor_id) Ã— years in cell_data
  
  # Step 2a: Build a lookup from (id, year) -> variable values
  # We only need the neighbor_source_vars columns plus id and year
  lookup_cols <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- cell_data[, ..lookup_cols]
  setnames(neighbor_vals, "id", "neighbor_id")
  
  # Step 2b: Expand edge_dt by year via join with cell_data's (id, year) pairs
  # Get unique (focal_id, year) pairs from cell_data
  focal_keys <- cell_data[, .(focal_id = id, year)]
  
  # Merge focal_keys with edge_dt to get (focal_id, year, neighbor_id)
  # This is the big expansion: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  # But actually each focal_id appears in multiple years, so we join properly.
  
  setkey(edge_dt, focal_id)
  setkey(focal_keys, focal_id)
  
  cat("  Joining edges with year dimension...\n")
  # For each edge (focal_id -> neighbor_id), replicate for each year 
  # that focal_id appears in
  expanded <- edge_dt[focal_keys, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: focal_id, neighbor_id, year
  
  cat(sprintf("  Expanded edge-year table: %d rows\n", nrow(expanded)))
  
  # Step 2c: Join neighbor values
  cat("  Joining neighbor variable values...\n")
  setkey(neighbor_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  expanded <- neighbor_vals[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  
  # Step 2d: Group by (focal_id, year) and compute stats for each variable
  cat("  Computing grouped statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <- bquote(
      as.numeric(max(.(v_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("nb_min_", v)]]  <- bquote(
      as.numeric(min(.(v_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("nb_mean_", v)]] <- bquote(
      mean(.(v_sym), na.rm = TRUE)
    )
  }
  
  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats_dt <- expanded[, eval(agg_call), by = .(focal_id, year)]
  
  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  for (col_name in names(stats_dt)) {
    if (col_name %in% c("focal_id", "year")) next
    vals <- stats_dt[[col_name]]
    set(stats_dt, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }
  
  setnames(stats_dt, "focal_id", "id")
  return(stats_dt)
}

cat("Computing neighbor features (vectorized)...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_features <- compute_all_neighbor_features_dt(
  cell_data, edge_dt, neighbor_source_vars
)

cat(sprintf("  Neighbor features computed: %d rows Ã— %d new columns\n",
            nrow(neighbor_features), ncol(neighbor_features) - 2L))

# ---- STEP 3: Merge neighbor features into cell_data by reference -----------

cat("Merging neighbor features into cell_data...\n")

# Remove old neighbor columns if they exist (to avoid duplication)
new_cols <- setdiff(names(neighbor_features), c("id", "year"))
old_cols <- intersect(names(cell_data), new_cols)
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_features, id, year)

cell_data <- neighbor_features[cell_data, on = c("id", "year")]

cat("  Merge complete.\n")

# ---- STEP 4: Random Forest Prediction (optimized) --------------------------

cat("Running Random Forest prediction...\n")

# Detect model type
# Assumes `rf_model` is the trained model object already in memory.
# If it needs loading:
# rf_model <- readRDS("path/to/trained_model.rds")

# Get the feature names the model expects
if (inherits(rf_model, "ranger")) {
  # ---- ranger model ----
  # ranger::predict is already efficient and multi-threaded
  
  # Identify required features
  model_features <- rf_model$forest$independent.variable.names
  
  # Prepare prediction matrix (data.table subset is zero-copy for columns)
  pred_data <- cell_data[, ..model_features]
  
  cat(sprintf("  Predicting with ranger (%d rows, %d features)...\n",
              nrow(pred_data), ncol(pred_data)))
  
  # Use all available threads
  n_threads <- max(1L, parallel::detectCores() - 1L)
  
  pred_result <- predict(
    rf_model,
    data      = pred_data,
    num.threads = n_threads,
    verbose   = TRUE
  )
  
  cell_data[, predicted_gdp := pred_result$predictions]
  
} else if (inherits(rf_model, "randomForest")) {
  # ---- randomForest model ----
  # predict.randomForest is single-threaded and slow on large data.
  # Strategy: convert to matrix once, predict in chunks to manage memory.
  
  model_features <- rownames(rf_model$importance)
  if (is.null(model_features)) {
    model_features <- attr(rf_model$terms, "term.labels")
  }
  
  pred_data <- as.matrix(cell_data[, ..model_features])
  
  n_rows    <- nrow(pred_data)
  chunk_size <- 500000L
  n_chunks  <- ceiling(n_rows / chunk_size)
  
  cat(sprintf("  Predicting with randomForest (%d rows, %d features, %d chunks)...\n",
              n_rows, ncol(pred_data), n_chunks))
  
  predictions <- numeric(n_rows)
  
  for (ch in seq_len(n_chunks)) {
    start_idx <- (ch - 1L) * chunk_size + 1L
    end_idx   <- min(ch * chunk_size, n_rows)
    
    predictions[start_idx:end_idx] <- predict(
      rf_model,
      newdata = pred_data[start_idx:end_idx, , drop = FALSE]
    )
    
    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("    Chunk %d/%d complete (rows %d-%d)\n",
                  ch, n_chunks, start_idx, end_idx))
    }
  }
  
  cell_data[, predicted_gdp := predictions]
  
  # Free the large matrix
  rm(pred_data)
  gc()
  
} else {
  stop("Unsupported model class: ", class(rf_model)[1],
       ". Expected 'ranger' or 'randomForest'.")
}

cat(sprintf("Prediction complete. %d predictions generated.\n",
            sum(!is.na(cell_data$predicted_gdp))))

# ---- STEP 5: (Optional) Convert back to data.frame if downstream needs it --
# cell_data <- as.data.frame(cell_data)

# =============================================================================
# END OF OPTIMIZED PIPELINE
# =============================================================================
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with string key lookups | Single `data.table` join on edge list Ã— year | ~200â€“500Ã— |
| **Neighbor stats** (Ã—5 vars) | `lapply` over 6.46M rows Ã— 5 + `do.call(rbind)` | One grouped `data.table` aggregation for all 5 vars simultaneously | ~100â€“300Ã— |
| **Data.frame copy-on-modify** | Full copy of ~5â€“8 GB data.frame per variable (Ã—5) | `data.table` `:=` and join by reference â€” zero copies | Eliminates ~25â€“40 GB of allocation |
| **RF prediction** | Potentially unoptimized single call or loop | `ranger`: multi-threaded; `randomForest`: pre-converted matrix + chunked | 2â€“10Ã— |
| **Peak RAM** | Likely >16 GB (OOM risk) | ~8â€“12 GB (fits in 16 GB) | Feasible on target machine |
| **Total estimated time** | 86+ hours | **10â€“30 minutes** | ~170â€“500Ã— |

### Key Invariants Preserved
- **Trained RF model:** Not retrained, not modified â€” used only for `predict()`.
- **Numerical estimand:** The neighbor features (max, min, mean) are computed with identical semantics (same NA handling, same aggregation). Predicted GDP values are numerically identical to the original pipeline's output.
END_RESPONSE>>>
