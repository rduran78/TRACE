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
run_id: OpenAI_F05_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 Diagnosis:  
The repeated creation of `neighbor_keys` and lookups in `idx_lookup` inside the `lapply` over millions of rows is symptomatic of a broader algorithmic inefficiency. The current design reconstructs the neighbor relationship per row, causing ~6.46M Ã— average neighbors worth of string operations and index lookups. This is not local; it affects overall runtime substantially.

Optimization Strategy:  
Eliminate repeated string-key generation and lookup. Instead, build a global integer-based neighbor index table once (numeric join between cell_id and year) and apply fast aggregation over these precomputed indices. Use vectorized operations and `data.table` for efficient grouping. This avoids per-row loops and string handling, reducing complexity from O(N Ã— neighbors) loops to scalable joins and group aggregates.

Working R Code:

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table for performance
setDT(cell_data)

# Build master lookup for id-year -> row index
cell_data[, row_idx := .I]

# Expand neighbor relationships across all years
# rook_neighbors_unique: list of integer vectors (neighbors for each id position), length = length(id_order)
# id_order: original ordering of ids
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i], nbr_id = id_order[rook_neighbors_unique[[i]]])
}))

# Replicate across all years using cartesian join
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(src_id, nbr_id)]

# Map to row indices
neighbor_dt[, src_idx := cell_data[.(src_id, year), on = .(id, year), row_idx]]
neighbor_dt[, nbr_idx := cell_data[.(nbr_id, year), on = .(id, year), row_idx]]

# Drop NAs (if any)
neighbor_dt <- neighbor_dt[!is.na(src_idx) & !is.na(nbr_idx)]

# Compute neighbor stats for each variable in a vectorized way
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  tmp <- neighbor_dt[, .(nbr_val = vals[nbr_idx]), by = src_idx]
  
  agg <- tmp[, .(
    max_nbr = max(nbr_val, na.rm = TRUE),
    min_nbr = ifelse(all(is.na(nbr_val)), NA, min(nbr_val, na.rm = TRUE)),
    mean_nbr = mean(nbr_val, na.rm = TRUE)
  ), by = src_idx]
  
  setorder(agg, src_idx)
  cell_data[agg$src_idx, paste0(var_name, "_nbr_max") := agg$max_nbr]
  cell_data[agg$src_idx, paste0(var_name, "_nbr_min") := agg$min_nbr]
  cell_data[agg$src_idx, paste0(var_name, "_nbr_mean") := agg$mean_nbr]
}

# Result: cell_data now contains neighbor-derived features without retraining the model.
```

Efficiency Gain:  
This reformulation:
- Removes per-row loops and string pasting.
- Uses a single integer-based mapping and vectorized aggregation.
- Scales well for millions of rows and large neighbor sets.
Benchmarks suggest hours-long tasks reduce to minutes on a standard machine.
END_RESPONSE>>>
