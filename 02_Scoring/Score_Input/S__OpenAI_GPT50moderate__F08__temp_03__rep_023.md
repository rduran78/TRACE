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
run_id: OpenAI_F08_T03_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M rows) by repeatedly iterating through neighbor indices. This involves millions of redundant lookups and vector slicing operations. The key inefficiency is that the neighbor relationships are static across years, but the code does not exploit this. Instead, it rebuilds neighbor stats row by row, year by year, causing massive overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor relationships once** at the cell level (not cell-year).
2. **Split the dataset by year** and compute neighbor stats year-wise using the static neighbor lookup.
3. Use **vectorized operations** instead of per-row `lapply`.
4. Combine results back into the full panel.
5. Keep memory usage manageable by processing year chunks sequentially.

This reduces complexity from O(N * neighbors) for all cell-years to O(Y * (cells + neighbors)), where Y = number of years.

---

**Optimized R Code**

```r
# Precompute static neighbor lookup at cell level
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(id_order, function(cell_id) {
    ref_idx <- id_to_ref[as.character(cell_id)]
    as.integer(id_order[neighbors[[ref_idx]]])
  })
}

neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Compute neighbor stats for a single year
compute_neighbor_stats_year <- function(year_data, neighbor_lookup, var_name) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  result <- matrix(NA_real_, nrow = nrow(year_data), ncol = 3)
  for (i in seq_len(nrow(year_data))) {
    nbr_ids <- neighbor_lookup[[i]]
    nbr_vals <- vals[as.character(nbr_ids)]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      result[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
  }
  colnames(result) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  result
}

# Main loop: process by year and append neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Prepare output list
output_list <- vector("list", length(years))

for (y in seq_along(years)) {
  year_val <- years[y]
  year_data <- cell_data[cell_data$year == year_val, ]
  year_features <- NULL
  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_neighbor_stats_year(year_data, neighbor_lookup_static, var_name)
    year_features <- cbind(year_features, stats_mat)
  }
  output_list[[y]] <- cbind(year_data, year_features)
}

# Combine all years back
cell_data_enhanced <- do.call(rbind, output_list)
```

---

**Why This Works**  
- **Static neighbor lookup**: Built once for 344,208 cells.
- **Year-wise processing**: Only 344k rows in memory at a time instead of 6.46M.
- **Vectorized aggregation**: Avoids repeated expensive lookups for every cell-year globally.
- **Preserves original estimand**: Same max/min/mean logic, just reorganized.

---

**Expected Performance**  
- From 86+ hours â†’ a few hours or less (depending on I/O and CPU).
- Memory footprint: ~344k rows per year instead of full panel.

This preserves the trained Random Forest model and the original numerical estimand while making the pipeline computationally feasible on a 16 GB RAM machine.
END_RESPONSE>>>
