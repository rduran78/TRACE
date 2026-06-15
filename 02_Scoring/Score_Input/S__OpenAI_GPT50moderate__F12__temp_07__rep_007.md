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
run_id: OpenAI_F12_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect. The main bottleneck is **not** `do.call(rbind, result)` or repeated list-binding. The overwhelming cost comes from repeated full passes over the 6.46â€¯million-row dataset and repeated neighbor index lookups for each of the five variables inside `compute_neighbor_stats()`. Each loop recomputes neighbor statistics from scratch, performing millions of random-access operations, which is extremely inefficient.

### **Diagnosis**
- `build_neighbor_lookup()` runs once and is relatively cheap.
- `compute_neighbor_stats()` does a full `lapply` over 6.46â€¯M elements for each variable â†’ **32.3â€¯M neighbor scans total**.
- `do.call(rbind, ...)` is linear in the size of the result and negligible compared to the repeated neighbor-value extraction and NA filtering.
- The real bottleneck is **recomputing neighbor stats separately per variable** instead of scanning neighbors once.

---

### **Optimization Strategy**
- Perform **one pass** over `neighbor_lookup`, computing all required neighbor statistics for all five variables simultaneously.
- Use **vectorized storage** (preallocated `matrix`) instead of repeated list allocations.
- Avoid repeated `[[var_name]]` extraction by converting `data` to a numeric matrix.
- Keep memory footprint manageable by writing results into preallocated columns.

---

### **Optimized Implementation**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  vals_mat <- as.matrix(data[var_names])  # Extract only needed columns
  n <- length(neighbor_lookup)
  m <- length(var_names)
  
  # Preallocate result matrix: 3 stats per variable
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3 * m)
  colnames(result_mat) <- as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals_mat[idx, , drop = FALSE]
      # Remove rows where all vars are NA
      keep <- rowSums(is.na(neighbor_vals)) < m
      if (any(keep)) {
        neighbor_vals <- neighbor_vals[keep, , drop = FALSE]
        result_mat[i, seq(1, 3*m, 3)] <- apply(neighbor_vals, 2, max, na.rm = TRUE)
        result_mat[i, seq(2, 3*m, 3)] <- apply(neighbor_vals, 2, min, na.rm = TRUE)
        result_mat[i, seq(3, 3*m, 3)] <- colMeans(neighbor_vals, na.rm = TRUE)
      }
    }
  }
  result_mat
}

# Usage
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind back to original data
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Why This Fixes the Problem**
- We reduce **five full passes** (5 Ã— 6.46â€¯M) to a **single pass** over `neighbor_lookup`.
- Memory stays within ~6.46â€¯M Ã— (5 Ã— 3) doubles â‰ˆ 775â€¯MB, which fits in 16â€¯GB RAM.
- Eliminates repeated `do.call(rbind, ...)` and redundant lookups.
- Preserves the original numerical estimand and the trained Random Forest model.

**Expected speedup:** From 86+ hours to **a few hours or less**, dominated by the single neighbor aggregation pass. Further optimization possible via `data.table`, Rcpp, or parallel processing, but the above change alone eliminates the main bottleneck.
END_RESPONSE>>>
