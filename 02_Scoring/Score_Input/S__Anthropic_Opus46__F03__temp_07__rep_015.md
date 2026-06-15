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
run_id: Anthropic_F03_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the most critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`), `paste` to build keys, and named-vector indexing into `idx_lookup`.
  - Named-vector lookups in R are **hash-table lookups per element**, but doing this millions of times in an interpreted loop is extremely slow.
- The resulting `neighbor_lookup` is a **list of 6.46 million integer vectors** â€” enormous memory overhead from list metadata alone.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M elements, each subsetting a numeric vector, removing NAs, and computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3 million interpreted iterations total.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (repeated allocations).

**Combined cost:** The nested interpreted R loops over millions of rows, with per-row string operations and list allocations, easily explain the 86+ hour runtime.

### B. Prediction Workflow Bottlenecks (Random Forest Inference)

- With ~6.46M rows Ã— 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object can be memory-intensive.
- If `randomForest` (Breiman's R package) is used, `predict.randomForest` is known to be **slow on large data** â€” it copies the entire data frame internally and loops in R/C in a less optimized way than `ranger`.
- If the model is loaded from disk each time or if the prediction data frame is copied repeatedly, that adds overhead.
- Predicting all 6.46M rows in one call may cause memory spikes (16 GB laptop).

---

## 2. Optimization Strategy

| Component | Problem | Solution |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops in interpreted R loop | Replace with vectorized `data.table` merge/join â€” build all neighbor-row indices in one bulk operation |
| `compute_neighbor_stats` | Per-row lapply Ã— 5 variables | Replace with `data.table` grouped aggregation: explode neighbor pairs, join values, aggregate with `max/min/mean` by row |
| `neighbor_lookup` (list of 6.46M vectors) | Memory-heavy, slow to iterate | Eliminate entirely â€” use a flat two-column edge table (`row_i`, `neighbor_row_j`) |
| `do.call(rbind, ...)` | Slow list-to-matrix conversion | Unnecessary once using `data.table` grouped aggregation |
| RF prediction | Possible `randomForest` package slowness; memory spike | Batch prediction in chunks; if model is `randomForest`, convert to `ranger` format or predict in chunks; ensure single `predict()` call with no unnecessary copies |
| Data copying | `cell_data` reassigned in loop (may copy entire data frame each iteration) | Use `data.table` set-by-reference (`:=`) to add columns in-place |

### Key Principles
1. **Vectorize everything** â€” no row-level `lapply` over millions of rows.
2. **Use `data.table`** for joins and grouped aggregation (C-level, cache-friendly).
3. **Flat edge table** instead of a list-of-vectors neighbor lookup.
4. **In-place column addition** (`:=`) to avoid copying a wide data frame.
5. **Chunked prediction** to stay within 16 GB RAM.
6. **Preserve the trained model object and numerical output exactly.**

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Feature Preparation + Random Forest Prediction
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- Step 1: Build flat edge table (replaces build_neighbor_lookup) ---------
# This is the single most important optimization: we replace ~6.46M lapply
# iterations with a vectorized bulk join.

build_neighbor_edge_table <- function(dt, id_order, nb_object) {
  # nb_object: spdep nb list â€” nb_object[[i]] gives neighbor indices into
  # id_order for the i-th element of id_order.
  
  # 1a. Expand the nb object into a flat (source_id, neighbor_id) table.
  #     This has ~1.37M rows (directed rook-neighbor relationships).
  n_cells <- length(id_order)
  lens <- lengths(nb_object)
  from_idx <- rep(seq_len(n_cells), lens)
  to_idx   <- unlist(nb_object, use.names = FALSE)
  
  cell_edges <- data.table(
    source_id   = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  
  # 1b. Get unique years
  years <- sort(unique(dt$year))
  
  # 1c. Cross-join edges Ã— years to get the full set of
  #     (source_id, year, neighbor_id) triples.
  #     ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in memory.
  cell_edges_yr <- cell_edges[, CJ(year = years), by = .(source_id, neighbor_id)]
  # More memory-efficient: use a cross join then merge
  # Actually, CJ inside by is inefficient for large groups. Better approach:
  
  year_dt <- data.table(year = years)
  cell_edges_yr <- cell_edges[, .(source_id, neighbor_id)]
  cell_edges_yr <- cell_edges_yr[
    rep(seq_len(.N), each = length(years))
  ]
  cell_edges_yr[, year := rep(years, times = nrow(cell_edges))]
  
  # 1d. Add row indices for the source rows (for writing results back).
  #     We key dt by (id, year) and do an equi-join.
  dt[, row_idx := .I]
  
  # Source row index
  setkey(dt, id, year)
  cell_edges_yr[, c("src_row") := dt[.(source_id, year), row_idx, mult = "first"]]
  
  # Neighbor row index
  cell_edges_yr[, c("nbr_row") := dt[.(neighbor_id, year), row_idx, mult = "first"]]
  
  # Drop edges where either source or neighbor row doesn't exist
  cell_edges_yr <- cell_edges_yr[!is.na(src_row) & !is.na(nbr_row)]
  
  setkey(cell_edges_yr, src_row)
  
  return(cell_edges_yr)
}

message("Building neighbor edge table...")
edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))


# ---- Step 2: Compute & add neighbor features (replaces compute_neighbor_stats
#              + the outer for-loop) ------------------------------------------
# For each variable, we look up neighbor values via the edge table, then
# aggregate (max, min, mean) grouped by src_row.  Results are written back
# into cell_data by reference.

compute_and_add_all_neighbor_features <- function(dt, edge_dt,
                                                   neighbor_source_vars) {
  for (var_name in neighbor_source_vars) {
    message(sprintf("  Computing neighbor stats for: %s", var_name))
    
    # Pull neighbor values into the edge table
    vals <- dt[[var_name]]
    edge_dt[, nbr_val := vals[nbr_row]]
    
    # Aggregate: grouped by src_row, compute max/min/mean ignoring NAs
    agg <- edge_dt[!is.na(nbr_val),
                   .(v_max  = max(nbr_val),
                     v_min  = min(nbr_val),
                     v_mean = mean(nbr_val)),
                   keyby = src_row]
    
    # Prepare NA-filled result vectors (for rows with no valid neighbors)
    n <- nrow(dt)
    col_max  <- rep(NA_real_, n)
    col_min  <- rep(NA_real_, n)
    col_mean <- rep(NA_real_, n)
    
    # Fill in computed values
    rows <- agg$src_row
    col_max[rows]  <- agg$v_max
    col_min[rows]  <- agg$v_min
    col_mean[rows] <- agg$v_mean
    
    # Write columns by reference (no data-frame copy)
    max_name  <- paste0("neighbor_max_", var_name)
    min_name  <- paste0("neighbor_min_", var_name)
    mean_name <- paste0("neighbor_mean_", var_name)
    
    set(dt, j = max_name,  value = col_max)
    set(dt, j = min_name,  value = col_min)
    set(dt, j = mean_name, value = col_mean)
  }
  
  # Clean up temporary column
  edge_dt[, nbr_val := NULL]
  
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features...")
compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
message("Neighbor features complete.")

# Remove helper column
cell_data[, row_idx := NULL]


# ---- Step 3: Optimized Random Forest Prediction ----------------------------

predict_in_chunks <- function(model, newdata, chunk_size = 500000L) {
  # Predicts in chunks to control peak memory on a 16 GB laptop.
  # Works with ranger, randomForest, or any model with a predict() method.
  
  n <- nrow(newdata)
  
  if (n <= chunk_size) {
    # Small enough to predict in one pass
    return(predict(model, data = newdata)$predictions %||%
           predict(model, newdata = newdata))
  }
  
  # Determine the predict interface
  is_ranger <- inherits(model, "ranger")
  
  # Pre-allocate result vector
  preds <- numeric(n)
  
  starts <- seq(1L, n, by = chunk_size)
  
  for (k in seq_along(starts)) {
    i_start <- starts[k]
    i_end   <- min(i_start + chunk_size - 1L, n)
    chunk   <- newdata[i_start:i_end, ]
    
    if (is_ranger) {
      p <- predict(model, data = chunk)$predictions
    } else {
      # randomForest or other
      p <- predict(model, newdata = chunk)
    }
    
    preds[i_start:i_end] <- p
    
    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("    Predicted %s / %s rows",
                      format(i_end, big.mark = ","),
                      format(n, big.mark = ",")))
    }
  }
  
  return(preds)
}

# Load the trained model (once)
message("Loading trained Random Forest model...")
rf_model <- readRDS("path/to/trained_rf_model.rds")  # <-- adjust path

# Prepare the prediction matrix: select only the predictor columns the model
# expects, in the correct order.  Avoid copying the full data.table.
if (inherits(rf_model, "ranger")) {
  predictor_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names differently
  predictor_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Subset to predictors only (data.table subsetting is memory-efficient)
pred_data <- cell_data[, ..predictor_names]

message("Running predictions...")
cell_data[, predicted_gdp := predict_in_chunks(rf_model, pred_data,
                                                chunk_size = 500000L)]
message("Prediction complete.")

# Clean up large temporaries
rm(pred_data, edge_dt)
gc()
```

---

## Summary of Expected Speedup

| Stage | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | Single vectorized `data.table` join | **~100â€“500Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M `lapply` iterations + `do.call(rbind)` | 5 grouped `data.table` aggregations (C-level) | **~50â€“200Ã—** |
| Column assignment | Full `cell_data` copy per iteration (wide df) | `set()` by reference â€” zero copies | **~5â€“10Ã—** |
| RF prediction | Possible single monolithic call (memory spike) | Chunked, single model load, correct `predict` dispatch | **~1â€“3Ã—** (memory-safe) |
| **Overall** | **~86+ hours** | **Estimated ~10â€“30 minutes** | **~150â€“500Ã—** |

The trained Random Forest model is never modified, and the numerical predictions (the estimand) are identical â€” only the data preparation and prediction workflow are optimized.
END_RESPONSE>>>
