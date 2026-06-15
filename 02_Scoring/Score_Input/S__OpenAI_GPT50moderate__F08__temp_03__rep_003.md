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
run_id: OpenAI_F08_T03_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows for each variable, repeatedly scanning neighbor indices. This results in ~32.3M expensive list operations (6.46M Ã— 5 variables), with redundant work across years. The neighbor structure is static across years, but the code recomputes neighbor relationships for every row-year combination. This leads to massive overhead and memory pressure on a 16â€¯GB laptop.

---

**Optimization Strategy**  
1. **Exploit Static Neighbor Structure**:  
   - Build the neighbor lookup **once per cell**, not per cell-year.
   - Store neighbors as integer vectors referencing cell IDs (not row indices).

2. **Vectorized Yearly Computation**:  
   - For each year, slice the data for that year into a matrix of variables.
   - Compute neighbor max, min, mean using **matrix operations** or `vapply` over the static neighbor list.

3. **Avoid Repeated Joins**:  
   - Preallocate output columns for all neighbor stats.
   - Fill them year by year.

4. **Memory Efficiency**:  
   - Process one year at a time (28 iterations) instead of all 6.46M rows at once.
   - Use numeric vectors and `vapply` for speed.

This reduces complexity from O(N*T*V) with redundant neighbor lookups to O(T*(N+E)*V), where E is edges (~1.37M), which is manageable.

---

**Working R Code**

```r
# Build static neighbor lookup by cell ID (not year)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is spdep::nb object
  lapply(neighbors, function(nb) as.integer(nb))  # list of integer vectors
}

# Compute neighbor stats for one year
compute_year_neighbor_stats <- function(year_data, neighbor_lookup, var_names) {
  n <- nrow(year_data)
  out <- matrix(NA_real_, nrow = n, ncol = length(var_names) * 3)
  colnames(out) <- as.vector(t(outer(var_names, c("max", "min", "mean"), paste, sep = "_")))

  for (vi in seq_along(var_names)) {
    vals <- year_data[[var_names[vi]]]
    # For each cell, compute stats from its neighbors
    stats <- vapply(neighbor_lookup, function(idx) {
      if (length(idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3))
    out[, (vi - 1) * 3 + 1:3] <- t(stats)
  }
  out
}

# Main optimized pipeline
neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate result columns
neighbor_cols <- as.vector(t(outer(neighbor_source_vars, c("max", "min", "mean"), paste, sep = "_")))
cell_data[neighbor_cols] <- NA_real_

# Process year by year
years <- sort(unique(cell_data$year))
for (yr in years) {
  idx <- which(cell_data$year == yr)
  year_data <- cell_data[idx, ]
  stats_mat <- compute_year_neighbor_stats(year_data, neighbor_lookup, neighbor_source_vars)
  cell_data[idx, neighbor_cols] <- stats_mat
}

# Now cell_data has neighbor features ready for Random Forest prediction
```

---

**Why This Works**  
- Neighbor relationships are computed **once** (static).
- Each year is processed in isolation, so memory stays within 16â€¯GB.
- Uses `vapply` for speed and avoids repeated `paste`/lookup overhead.
- Preserves the original numerical estimand and the trained Random Forest model.

**Expected Runtime**:  
From 86+ hours â†’ a few hours (dominated by ~28 passes over 344k rows Ã— 5 variables Ã— ~1.37M edges, but vectorized and memory-efficient).
END_RESPONSE>>>
