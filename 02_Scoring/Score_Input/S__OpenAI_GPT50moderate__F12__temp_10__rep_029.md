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
run_id: OpenAI_F12_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that the main bottleneck is `do.call(rbind, result)` (and repeated list-binding inside `compute_neighbor_stats`) is incorrect.  
`rbind` over ~6.46M rows does have cost, but the deeper bottleneck comes from the **inner `lapply` per variable over the entire neighbor list**. Each iteration repeatedly computes neighbor stats for 6.46M rows by extracting indices and subsetting `vals`. With 5 variables, that's â‰ˆ32M neighbor lookups and allocationsâ€”**the dominant cost**. This design scales poorly because it recalculates neighbor statistics for every variable instead of reusing the neighbor structure.

---

### Correct Optimization Strategy
- Precompute neighbor indices **once** in an efficient structure (already done with `neighbor_lookup`).
- Consolidate computations for all 5 variables in **one pass** through `neighbor_lookup`.
- Avoid repeated subsetting and repeated traversals by operating on a **matrix view** of `(row Ã— 5 vars)` values.
- Output a matrix of shape `(nrow(data) Ã— (5 Ã— 3))` for max, min, mean for each var in one sweep.

This removes 5 full passes (and 5Ã—6.46M loops).

---

### Optimized R Code

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  # Pre-extract matrix of relevant variables
  vals_mat <- as.matrix(data[, var_names, drop = FALSE])
  n <- length(neighbor_lookup)
  k <- length(var_names)
  
  # Result matrix: rows = nrow(data), cols = 3 summaries (max,min,mean) * k variables
  result <- matrix(NA_real_, nrow = n, ncol = k * 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    
    neigh_vals <- vals_mat[idx, , drop = FALSE]
    neigh_vals[is.na(neigh_vals)] <- NA  # keep NA filtering explicit
    if (!all(is.na(neigh_vals))) {
      col_max <- apply(neigh_vals, 2, max, na.rm = TRUE)
      col_min <- apply(neigh_vals, 2, min, na.rm = TRUE)
      col_mean <- apply(neigh_vals, 2, mean, na.rm = TRUE)
      result[i, ] <- c(rbind(col_max, col_min, col_mean)) # interleave stats
    }
  }
  
  colnames(result) <- as.vector(sapply(var_names, function(v)
    paste0(v, c("_max", "_min", "_mean"))))
  
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind to main data:
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### Why This Works:
- Reduces **5 full passes â†’ 1 pass** over 6.46M rows and neighbor lists.
- Eliminates `do.call(rbind, ...)` per variable.
- Converts repeated subsetting into a single matrix-backed operation.
- Memory footprint is predictable and significantly faster because of vectorization.

---

**Expected Impact:**  
Empirical benchmarks show this approach typically improves performance by **>10Ã—** for similar spatial panel datasets, reducing runtime from 86+ hours to a more manageable range (often minutes to a few hours on 16â€¯GB RAM).  

**Trained model and estimand are preserved**â€”only feature computation is optimized.
END_RESPONSE>>>
