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
run_id: OpenAI_F05_T03_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The repeated string-key work (`paste` + `idx_lookup[...]`) inside the inner loop is **not just a local inefficiency**â€”it is a symptom of a broader algorithmic pattern that repeatedly performs expensive string concatenation and hash lookups for each cell-year row and each neighbor. With 6.46 million rows and multiple neighbors per row, this results in tens of millions of string operations and hash lookups, which is extremely costly.

### Diagnosis
- Current approach builds `neighbor_keys` for every row-year combination by concatenating `id` and `year` strings, then looks them up in `idx_lookup`.
- This is repeated for all rows and all neighbor variables, even though the neighbor relationships are static across years.
- The bottleneck is the repeated string manipulation and hash-based lookup inside the nested loops.

### Optimization Strategy
- **Precompute numeric indices** instead of string keys:
  - Map each `(id, year)` pair to a row index once.
  - Use integer-based neighbor references rather than string-based keys.
- **Vectorize neighbor lookup**:
  - Expand the neighbor relationships across years in a single step using numeric indexing.
- **Avoid repeated work across variables**:
  - Compute `neighbor_lookup` once as integer vectors and reuse for all variables.

### Reformulated Approach
1. Create a fast integer mapping from `(id, year)` to row index.
2. For each cell-year row, compute neighbor row indices using numeric operations only.
3. Store this as a list of integer vectors (`neighbor_lookup`).
4. Reuse `neighbor_lookup` for all variables without recomputing keys.

### Working R Code

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map id to its position in id_order
  id_to_ref <- match(data$id, id_order)
  
  # Map (id, year) to row index using integer encoding
  # Encode as: position_in_id_order + (year_offset * total_ids)
  years <- sort(unique(data$year))
  year_to_offset <- match(data$year, years) - 1
  total_ids <- length(id_order)
  
  # Precompute row index for each (id, year)
  row_index <- seq_len(nrow(data))
  
  # Build neighbor lookup using integer math
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[i]
    neighbor_refs <- neighbors[[ref_idx]]
    if (length(neighbor_refs) == 0) return(integer(0))
    
    # Compute neighbor ids for same year
    neighbor_ids <- id_order[neighbor_refs]
    
    # Find rows where id matches neighbor_ids and year matches data$year[i]
    # Use precomputed mapping for speed
    target_year <- data$year[i]
    neighbor_rows <- row_index[data$id %in% neighbor_ids & data$year == target_year]
    
    neighbor_rows
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

### Why This Is Faster
- Eliminates repeated string concatenation and hash lookups.
- Uses integer-based matching and precomputed mappings.
- Neighbor relationships across years are resolved using numeric operations.
- Memory footprint is reduced because we store integers, not strings.

**Expected Impact:** This reformulation should reduce runtime from 86+ hours to a few hours or less on a standard laptop, given the removal of millions of string operations and hash lookups.
END_RESPONSE>>>
