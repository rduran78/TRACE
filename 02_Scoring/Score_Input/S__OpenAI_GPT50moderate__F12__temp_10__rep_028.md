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
run_id: OpenAI_F12_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that `do.call(rbind, result)` and repeated list binding are the main bottlenecks is incorrect. These operations occur in `compute_neighbor_stats()`, but they are trivial compared to the *outer loop recomputation* and the repeated full traversal over 6.46 million rows for each variable (`neighbor_source_vars`). The true bottleneck is the repeated execution of neighbor lookups and aggregations across the entire panel for each variable, rather than batching the work.

**Deeper Bottleneck:**  
- `compute_neighbor_stats()` is executed 5 times (once for each variable), each time iterating through all rows.
- The neighbor lookup (`neighbor_lookup`) itself is static across variables and could be leveraged for vectorized computation.
- Building and binding lists repeatedly amplifies the problem across 6.46M rows Ã— 5 variables.

**Optimization Strategy:**  
Compute **all required neighbor statistics for all variables in a single pass**, then bind results once. This eliminates redundant loops and avoids repeatedly calling functions that traverse millions of records. Use `vapply` for safer and faster aggregation, and preallocate final results rather than doing repeated bind operations.

---

### **Working R Code:**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  # Preallocate matrix: rows = nrow(data), cols = 3 * length(var_names)
  n <- length(neighbor_lookup)
  res_mat <- matrix(NA_real_, nrow = n, ncol = 3 * length(var_names))
  col_names <- unlist(lapply(var_names, function(v) c(paste0(v, "_max"),
                                                     paste0(v, "_min"),
                                                     paste0(v, "_mean"))))
  colnames(res_mat) <- col_names
  
  vals_list <- lapply(var_names, function(v) data[[v]])
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(vals_list)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      stats <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      res_mat[i, (3*j - 2):(3*j)] <- stats
    }
  }
  
  as.data.frame(res_mat)
}

# Usage
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_stats_df <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_vars)

# Bind to cell_data in one operation
cell_data <- cbind(cell_data, neighbor_stats_df)
```

---

### **Expected Gains:**
- Single pass through 6.46M rows instead of 5 passes.
- Avoids repeated `do.call(rbind, ...)` and repeated `lapply(...)` calls per variable.
- Reduces run time from multiple days (86+ hours) to a few hours or less, depending on machine and I/O constraints.
- Preserves original Random Forest model and estimands because only input features formatting changes; underlying statistics remain identical.  

This batching and preallocation approach exploits **loop fusion** and memory locality, addressing the true bottleneck efficiently.
END_RESPONSE>>>
