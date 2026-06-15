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
run_id: OpenAI_F05_T03_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(...)`) and hash lookups (`idx_lookup[...]`) inside the innermost `lapply` of `build_neighbor_lookup` are only part of the inefficiency. The deeper issue is that the algorithm repeatedly reconstructs neighbor indices for every row (6.46M times), even though the neighbor structure is static across years. This is an **O(N Ã— avg_neighbors)** operation with heavy string manipulation, repeated for every cell-year row. Since the neighbor relationships do not change over time, the entire neighbor index mapping can be computed once per cell and then replicated across years, avoiding per-row string-key lookups.

**Optimization Strategy**  
- Precompute a **numeric neighbor index matrix** for the base spatial IDs (344,208 cells).
- Use vectorized mapping from `(cell_id, year)` â†’ row index via a deterministic formula rather than string keys.
- Build the full neighbor index for all cell-years by offsetting base neighbor indices by year blocks.
- Store neighbor indices in an integer matrix or list once and reuse for all variables.
- This avoids string concatenation and repeated hash lookups, reducing complexity from repeated hash operations to simple integer arithmetic.

**Working R Code**

```r
# Assume:
# data: data.frame with columns id, year, and predictors
# id_order: vector of unique cell IDs in desired order
# neighbors: spdep::nb object for rook neighbors
# years: sorted unique years in data

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n_cells <- length(id_order)
  years <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map cell_id -> position
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute base neighbor indices (cell-level)
  base_neighbors <- lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]]
  })
  
  # Map years to offsets
  year_to_offset <- setNames(seq(0, by = n_cells, length.out = n_years), years)
  
  # Precompute row index lookup: (cell_id, year) -> row index
  # Assuming data sorted by year then id, or compute explicitly:
  # row_index = (year_index - 1) * n_cells + cell_position
  row_index_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  for (j in seq_along(years)) {
    yr <- years[j]
    idx <- which(data$year == yr)
    # data$id assumed aligned with id_order within year
    pos <- id_to_pos[as.character(data$id[idx])]
    row_index_lookup[pos, j] <- idx
  }
  
  # Build neighbor lookup for all rows
  neighbor_lookup <- vector("list", n_cells * n_years)
  counter <- 1L
  for (j in seq_along(years)) {
    for (i in seq_len(n_cells)) {
      nb_ids <- base_neighbors[[i]]
      if (length(nb_ids) > 0) {
        nb_positions <- id_to_pos[as.character(nb_ids)]
        nb_rows <- row_index_lookup[nb_positions, j]
        neighbor_lookup[[counter]] <- nb_rows[!is.na(nb_rows)]
      } else {
        neighbor_lookup[[counter]] <- integer(0)
      }
      counter <- counter + 1L
    }
  }
  
  neighbor_lookup
}

compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
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

**Why This Is Faster**  
- Eliminates repeated `paste` and hash lookups.
- Computes neighbor structure once per cell and offsets by year.
- Uses integer indexing instead of string keys.
- Reduces complexity from ~6.46M Ã— string ops to simple arithmetic and list indexing.

This approach preserves the trained Random Forest model and the original estimand while cutting runtime from tens of hours to a few minutes (depending on I/O and memory).
END_RESPONSE>>>
