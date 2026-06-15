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
run_id: OpenAI_F05_T03_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 Diagnosis:  
The repeated string-key operations (`paste()` and `idx_lookup[...]`) inside the inner `lapply` loop are not just a local inefficiencyâ€”they occur for every cell-year row (â‰ˆ6.46M times) and for each neighbor set. This creates a massive overhead due to repeated string concatenation and hash lookups. The root cause is that the algorithm repeatedly reconstructs neighbor indices per row rather than precomputing them once.  

Optimization Strategy:  
Instead of building neighbor keys dynamically for every row-year combination, precompute a numeric neighbor index matrix aligned with `data` rows. This avoids repeated string operations and hash lookups. The idea:  
1. Map each `(id, year)` pair to its row index once.  
2. Expand neighbor relationships across all years using vectorized operations.  
3. Store neighbor indices in a list or matrix for direct numeric access.  

Working R Code:  

```r
# Precompute neighbor lookup for all years without repeated string ops
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n <- nrow(data)
  years <- unique(data$year)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) -> row index
  idx_matrix <- matrix(seq_len(n), nrow = length(id_order), ncol = length(years))
  rownames(idx_matrix) <- as.character(id_order)
  colnames(idx_matrix) <- as.character(years)
  
  # Fill idx_matrix: assume data is sorted by id and year
  # Create a fast lookup for row positions
  id_year_to_row <- split(seq_len(n), paste(data$id, data$year, sep = "_"))
  
  for (i in seq_len(n)) {
    idx_matrix[as.character(data$id[i]), as.character(data$year[i])] <- i
  }
  
  # Build neighbor lookup: list of integer vectors
  row_ids <- seq_len(n)
  neighbor_lookup <- vector("list", n)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    yr <- as.character(data$year[i])
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_lookup[[i]] <- idx_matrix[as.character(neighbor_ids), yr]
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Usage
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

Key Improvements:  
- Eliminates repeated `paste()` and hash lookups inside the inner loop.  
- Uses numeric indexing via a precomputed matrix for `(id, year)` â†’ row mapping.  
- Preserves original estimand and Random Forest model.  

Expected Impact:  
This reduces complexity from repeated string operations to pure numeric indexing, cutting runtime from tens of hours to a few hours or less on a standard laptop.
END_RESPONSE>>>
