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
run_id: Anthropic_F03_T00_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector)
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations Ã— multiple string allocations and hash lookups against a 6.46M-entry named vector. This alone can take many hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix â€” this is slow because `rbind` on a list of millions of small vectors has quadratic-like overhead in practice.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 Ã— 6.46M = ~32.3M iterations total.

### 1.2 Prediction Bottleneck

With ~110 predictors and 6.46M rows, `predict.randomForest` (or `predict.ranger`) on the full dataset is a single large matrix operation. Key issues:
- If using the `randomForest` package, `predict()` is single-threaded and slow on large data.
- If the model object is large, loading it from disk and copying it in memory is expensive.
- Predicting all 6.46M rows at once may exceed RAM if the forest is large (many trees Ã— many nodes).

### 1.3 Summary of Time Sinks (estimated share of 86+ hours)

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~40-50% | Per-row string ops, named-vector lookups on 6.46M keys |
| `compute_neighbor_stats` (Ã—5) | ~25-35% | Per-row lapply, `do.call(rbind, ...)` on millions of rows |
| RF prediction | ~15-25% | Single-threaded predict, possible memory pressure |
| Model I/O & object copying | ~5% | Large serialized object |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorize with `data.table`

**Replace `build_neighbor_lookup`** with a fully vectorized join:
1. Expand the `nb` object into an edge-list (cell_id â†’ neighbor_cell_id) once.
2. Join with the panel data on (neighbor_cell_id, year) to get row indices of neighbors.
3. Group by the focal row and compute stats using `data.table` grouped aggregation â€” no per-row `lapply` needed.

This eliminates all 6.46M `paste`/lookup iterations and replaces them with a single indexed merge + grouped aggregation.

**Replace `compute_neighbor_stats`** with a single `data.table` grouped operation per variable (or all variables at once), computing max/min/mean in vectorized C code.

### 2.2 Prediction â€” Use `ranger` predict or chunk-based prediction

- If the model is a `ranger` object, `predict()` is already multi-threaded â€” just call it on the full dataset.
- If the model is a `randomForest` object, convert prediction to chunked batches to control memory, or (if feasible) re-export the model to `ranger`-compatible format. Since we **cannot retrain**, we keep the original model but chunk the prediction.
- Pre-allocate the prediction output vector; avoid copying the data frame.

### 2.3 Expected Speedup

| Component | Before | After (estimated) |
|---|---|---|
| Neighbor lookup + stats | ~60-70 hrs | ~2-5 min |
| RF prediction (6.46M rows) | ~15-20 hrs | ~10-40 min (chunked/parallel) |
| **Total** | **86+ hrs** | **~15-45 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Neighbor Feature Preparation + RF Prediction
# =============================================================================
# Requirements: data.table, ranger (if applicable), randomForest
# Preserves: trained RF model object, original numerical estimand

library(data.table)

# ---- STEP 0: Convert panel data to data.table ----
# Assumes: cell_data is a data.frame with columns: id, year, ntl, ec,
#          pop_density, def, usd_est_n2, ... (110 predictor columns)
# Assumes: rook_neighbors_unique is an nb object (list of integer index vectors)
# Assumes: id_order is the vector of cell IDs corresponding to nb indices
# Assumes: rf_model is the pre-trained Random Forest model (randomForest or ranger)

cat("Converting to data.table...\n")
dt <- as.data.table(cell_data)

# Preserve original row order for output
dt[, .row_order := .I]

# ---- STEP 1: Build edge list from nb object (one-time, vectorized) ----
cat("Building edge list from nb object...\n")

build_edge_list_dt <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for neighbors of id_order[i]
  # We expand this into a two-column data.table: focal_id, neighbor_id
  
  n <- length(nb_obj)
  
  # Count neighbors per cell (nb objects use 0L to indicate no neighbors)
  n_neighbors <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate
  focal_id    <- integer(total_edges)
  neighbor_id <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    ni <- n_neighbors[i]
    if (ni > 0L) {
      idx_range <- pos:(pos + ni - 1L)
      focal_id[idx_range]    <- id_order[i]
      neighbor_id[idx_range] <- id_order[nb_obj[[i]]]
      pos <- pos + ni
    }
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_dt)))

# ---- STEP 2: Compute all neighbor features via vectorized join + group-by ----
cat("Computing neighbor features...\n")

compute_all_neighbor_features <- function(dt, edge_dt, source_vars) {
  # Create a lookup: (id, year) -> row index in dt, plus source variable values
  # We only need id, year, and the source variables for the neighbor lookup
  
  lookup_cols <- c("id", "year", source_vars)
  lookup <- dt[, ..lookup_cols]
  
  # Join edge list with dt to get (focal_id, year) for each row,

  # then join with lookup on (neighbor_id, year) to get neighbor values.
  
  # Step A: For each row in dt, get its focal_id and year, then expand by edges
  # dt has columns: id, year, ...
  # edge_dt has columns: focal_id, neighbor_id
  
  # Create focal table: row_index, id, year
  focal <- dt[, .(row_idx = .row_order, focal_id = id, year = year)]
  
  # Merge focal with edge_dt on focal_id to get all (row_idx, year, neighbor_id) combos
  setkey(edge_dt, focal_id)
  setkey(focal, focal_id)
  
  # This is the big join: each focal row Ã— its neighbors
  # Result: one row per (focal_row, neighbor_cell, year)
  expanded <- edge_dt[focal, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # expanded has: focal_id, neighbor_id, row_idx, year
  
  cat(sprintf("  Expanded edge-row table: %d rows\n", nrow(expanded)))
  
  # Step B: Join with lookup to get neighbor variable values
  setnames(lookup, "id", "neighbor_id")
  setkeyv(lookup, c("neighbor_id", "year"))
  setkeyv(expanded, c("neighbor_id", "year"))
  
  merged <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # merged has: neighbor_id, year, <source_vars>, focal_id, row_idx
  
  cat("  Computing grouped statistics...\n")
  
  # Step C: Group by row_idx and compute max, min, mean for each source var
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- bquote(max(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("n_min_", v)]]  <- bquote(min(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("n_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats <- merged[, eval(agg_call), by = row_idx]
  
  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col_name in names(stats)) {
    if (col_name == "row_idx") next
    vals <- stats[[col_name]]
    stats[is.infinite(vals), (col_name) := NA_real_]
  }
  
  # Step D: Join stats back to dt by row_idx
  setkey(stats, row_idx)
  setkey(dt, .row_order)
  
  new_cols <- setdiff(names(stats), "row_idx")
  dt[stats, (new_cols) := mget(paste0("i.", new_cols)), on = .(.row_order = row_idx)]
  
  return(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

dt <- compute_all_neighbor_features(dt, edge_dt, neighbor_source_vars)

cat("Neighbor features complete.\n")

# Clean up large intermediate objects
rm(edge_dt)
gc()

# ---- STEP 3: Prepare prediction data ----
cat("Preparing prediction matrix...\n")

# Remove helper column before prediction
dt[, .row_order := NULL]

# Get the predictor variable names expected by the model
# (Adjust this depending on your model object type)
if (inherits(rf_model, "ranger")) {
  predictor_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names in the model
  predictor_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all predictors are present
missing_preds <- setdiff(predictor_names, names(dt))
if (length(missing_preds) > 0) {
  stop("Missing predictor columns: ", paste(missing_preds, collapse = ", "))
}

# ---- STEP 4: Chunked prediction to manage memory ----
cat("Running Random Forest prediction...\n")

predict_chunked <- function(model, dt, predictor_names, chunk_size = 500000L) {
  n <- nrow(dt)
  predictions <- numeric(n)
  n_chunks <- ceiling(n / chunk_size)
  
  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)
    
    cat(sprintf("  Predicting chunk %d/%d (rows %d-%d)...\n", 
                i, n_chunks, start_idx, end_idx))
    
    # Extract only predictor columns for this chunk (minimizes memory copy)
    chunk_df <- as.data.frame(dt[start_idx:end_idx, ..predictor_names])
    
    if (inherits(model, "ranger")) {
      pred <- predict(model, data = chunk_df, num.threads = parallel::detectCores())
      predictions[start_idx:end_idx] <- pred$predictions
    } else if (inherits(model, "randomForest")) {
      predictions[start_idx:end_idx] <- predict(model, newdata = chunk_df)
    }
    
    rm(chunk_df)
    # Trigger GC every few chunks to keep memory in check
    if (i %% 5 == 0) gc()
  }
  
  return(predictions)
}

dt[, predicted_gdp := predict_chunked(rf_model, dt, predictor_names)]

cat("Prediction complete.\n")

# ---- STEP 5: Convert back to data.frame if needed ----
cell_data <- as.data.frame(dt)

cat("Pipeline finished.\n")
```

---

## 4. KEY DESIGN DECISIONS AND NOTES

| Decision | Rationale |
|---|---|
| **Single edge-list expansion + grouped aggregation** instead of per-row `lapply` | Eliminates 6.46M R-level loop iterations; `data.table` grouped ops run in C |
| **All 5 variables aggregated in one pass** | One large join instead of 5 separate passes; reduces I/O and memory churn |
| **`allow.cartesian = TRUE`** in the join | Each focal row has multiple neighbors, so the join is intentionally many-to-many |
| **Chunked prediction (500K rows)** | Keeps peak memory well within 16 GB; each chunk's predictor matrix is ~500K Ã— 110 â‰ˆ 420 MB |
| **`num.threads` for ranger** | Exploits all cores for tree-level parallelism during inference |
| **Infinite â†’ NA replacement** | `max(na.rm=TRUE)` on an empty/all-NA vector returns `-Inf`; we map these back to `NA` to preserve the original estimand's NA semantics |
| **No model retraining** | The trained `rf_model` object is used as-is; only `predict()` is called |
| **Original numerical estimand preserved** | The same max/min/mean neighbor statistics are computed with identical arithmetic; predictions come from the same model with the same features |

### Expected Runtime

- **Neighbor features:** ~2â€“5 minutes (dominated by the cartesian join of ~6.46M rows Ã— ~4 neighbors â‰ˆ 26M expanded rows, then grouped aggregation).
- **RF prediction:** ~10â€“40 minutes depending on forest size and whether `ranger` (parallel) or `randomForest` (single-threaded) is used.
- **Total: ~15â€“45 minutes** vs. the original 86+ hours â€” roughly a **100â€“300Ã— speedup**.
END_RESPONSE>>>
