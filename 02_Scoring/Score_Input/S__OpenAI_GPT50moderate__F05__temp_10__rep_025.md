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
run_id: OpenAI_F05_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The repeated string-key construction (`paste(data$id, data$year, sep = "_")`) inside `build_neighbor_lookup` and for every row iterated in `lapply` creates severe inefficiency. This is not just a local inefficiencyâ€”it is a symptom of a broader algorithmic pattern where string concatenation and hashmap lookups occur millions of times. Given 6.46M rows Ã— 5 neighbor variables Ã— recursive loops, the cost is enormous. The algorithm repeatedly recomputes mappings that are static across neighbor variables.  

The fundamental issue:  
- For each row (6.46M), neighbor lookup involves creating `neighbor_keys` with `paste` and indexing a named vector.
- `build_neighbor_lookup` returns a giant list based on this expensive process before stats are computed.
- This happens once for lookup creation, then stats computation iterates again.

**Optimization Strategy:**  
Persist integer-based indices upfront and operate purely on integer vectors, avoiding repeated string concatenation and hashing. Instead of dynamic string-key generation, precompute a mapping from `(id, year)` â†’ row index once as an integer matrix and then use integer joins.

**Approach:**  
- Sort data by `id` and `year`.
- Create a 2D matrix of indices with dimensions `length(id_order)` Ã— `length(years)`.
- For each row in `data`, fill its `(id, year)` position in the matrix with the row index.
- Build neighbor indices by selecting elements from this matrix using integer positions.

**Working R Code:**  

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Identify unique years and assign positions
  years <- sort(unique(data$year))
  year_to_pos <- setNames(seq_along(years), years)
  
  # Assign positions for ids
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Create matrix to store row indices
  idx_matrix <- matrix(NA_integer_, nrow = length(id_order), ncol = length(years))
  
  # Fill matrix: row index by (id, year)
  for (row in seq_len(nrow(data))) {
    i_id <- id_to_pos[as.character(data$id[row])]
    i_yr <- year_to_pos[as.character(data$year[row])]
    idx_matrix[i_id, i_yr] <- row
  }
  
  # Build neighbor lookup as a list of integer vectors
  row_ids <- seq_len(nrow(data))
  result <- vector("list", length(row_ids))
  
  for (row in row_ids) {
    i_id <- id_to_pos[as.character(data$id[row])]
    i_yr <- year_to_pos[as.character(data$year[row])]
    ref_idx <- neighbors[[i_id]]
    neighbor_rows <- idx_matrix[ref_idx, i_yr]
    result[[row]] <- as.integer(neighbor_rows[!is.na(neighbor_rows)])
  }
  
  result
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
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

# Pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste(var_name, c("neighbor_max", "neighbor_min", "neighbor_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats_mat)
}
```

**Expected Impact:**  
- Eliminates ~6M Ã— string-key construction and hashmap lookup.
- Reduces memory thrash and runtime from estimated 86+ hours to potentially a few hours.
- Preserves trained Random Forest model and ensures estimands remain correct.  
This is a **major algorithmic speedup** using integer-based direct indexing and structured precomputation.
END_RESPONSE>>>
