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
run_id: OpenAI_F05_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key construction (`paste(data$id, data$year, sep="_")`) inside `build_neighbor_lookup` is not the only inefficiencyâ€”itâ€™s a symptom of a broader algorithmic pattern where, for every row (â‰ˆ6.46M), we repeatedly map neighbor IDs to row indices using string concatenation and hash lookups. This creates massive overhead in both memory and CPU. The lookups themselves are not necessary if we precompute a direct numeric mapping from `(id, year)` to row index.

**Optimization Strategy**  
Avoid repeated string concatenation and hashing by:
1. Precomputing a numeric index mapping from `id` and `year` to row positions using integer arrays rather than character keys.
2. Operating on integer vectors directly (no repeated calls to `paste` or `setNames`).
3. Leveraging vectorized operations and `vapply` or matrix-based joins.

We can build a `row_index_matrix` with dimensions `[id, year]` so that for any `(id, year)` we retrieve a row index in O(1) time without strings.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors, years) {
  n_ids <- length(id_order)
  n_years <- length(years)
  
  # Map ids to 1:n_ids
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  # Map years to 1:n_years
  year_to_ref <- setNames(seq_along(years), as.character(years))
  
  # Precompute row index matrix: rows = id, cols = year
  row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_index_matrix[cbind(id_to_ref[as.character(data$id)],
                          year_to_ref[as.character(data$year)])] <- seq_len(nrow(data))
  
  # Build neighbor lookup without string-based keys
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    yr_idx  <- year_to_ref[as.character(data$year[i])]
    neighbor_ref_ids <- neighbors[[ref_idx]]
    neighbor_rows <- row_index_matrix[neighbor_ref_ids, yr_idx]
    neighbor_rows[!is.na(neighbor_rows)]
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  
  t(result)  # return as matrix
}

# Pipeline
years <- sort(unique(cell_data$year))
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique, years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Gains**
- Eliminates repeated `paste()` and `setNames()` calls.
- Avoids hash-based lookups in favor of O(1) integer matrix indexing.
- Reduces runtime from **86+ hours to a few hours (or less)** depending on disk I/O and memory constraints.
- Preserves the trained Random Forest model and all numerical estimands.

This approach is memory-efficient for a 16 GB laptop:  
`row_index_matrix` = (344,208 ids Ã— 28 years) â‰ˆ 9.6M integers (~38MB). Very feasible.
END_RESPONSE>>>
