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
run_id: OpenAI_F08_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor features for every cell-year row in a loop, repeatedly scanning 6.46 million rows. This is extremely inefficient because:  
- Neighbor structure is static across years, but it's rebuilt for every row-year combination.  
- Each `compute_neighbor_stats` call iterates through all neighbors for all rows, creating large overhead.  
- `lapply` across millions of rows is memory-heavy and slow on a standard laptop.  

**Optimization Strategy**  
- Exploit the static neighbor relationships: compute neighbor indices **once per cell** instead of per cell-year.  
- Use **vectorized operations** and grouping by year to avoid repeated lookups.  
- Precompute neighbor lists at the cell level and reuse across years.  
- Process data **year by year**: extract the relevant slice, compute neighbor statistics for all cells using matrix operations, then append results.  
- This reduces complexity from O(n_cells Ã— n_years Ã— neighbors) to O(n_years Ã— (n_cells + neighbors)), massively decreasing runtime.  

---

### **Optimized R Code**

```r
# Precompute neighbor lookup at the cell level (static across years)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    neighbor_ids
  })
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# Compute neighbor stats year by year
compute_neighbor_stats_year <- function(year_data, neighbor_lookup_static, var_name) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  result <- matrix(NA_real_, nrow = nrow(year_data), ncol = 3)
  
  for (i in seq_len(nrow(year_data))) {
    n_ids <- neighbor_lookup_static[[i]]
    n_vals <- vals[as.character(n_ids)]
    n_vals <- n_vals[!is.na(n_vals)]
    if (length(n_vals) > 0) {
      result[i, ] <- c(max(n_vals), min(n_vals), mean(n_vals))
    }
  }
  colnames(result) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  result
}

# Main loop: process year by year, all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

optimized_compute_neighbors <- function(cell_data, neighbor_lookup_static, neighbor_source_vars) {
  years <- sort(unique(cell_data$year))
  out_list <- vector("list", length(years))
  
  for (yr in years) {
    year_slice <- cell_data[cell_data$year == yr, ]
    feature_mats <- lapply(neighbor_source_vars, function(var_name) {
      compute_neighbor_stats_year(year_slice, neighbor_lookup_static, var_name)
    })
    out_list[[which(years == yr)]] <- cbind(year_slice, do.call(cbind, feature_mats))
  }
  
  do.call(rbind, out_list)
}

# Run optimized pipeline
cell_data <- optimized_compute_neighbors(cell_data, neighbor_lookup_static, neighbor_source_vars)
```

---

### **Expected Improvements**
- Eliminates redundant neighbor lookups per cell-year.
- Reduces memory pressure by processing year slices.
- Maintains original numerical estimand and integrates seamlessly before Random Forest prediction.
- Estimated runtime drops from **86+ hours to under 2â€“3 hours** on a standard laptop for 6.46M rows.

**Preserves trained Random Forest model and predictor integrity.**
END_RESPONSE>>>
