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
run_id: OpenAI_F12_T00_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F12_OpenAI_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottlenecks is incorrect. While `rbind` on millions of rows is non-trivial, the deeper bottleneck lies in the repeated `lapply` over 6.46 million rows for each of 5 variables, performing neighbor lookups and aggregations in pure R loops. This results in ~32 million iterations and heavy memory churn. The real issue is the lack of vectorization and the repeated traversal of neighbor lists.

---

**Optimization Strategy:**  
- Precompute a long-format neighbor table (cell-year â†’ neighbor-year) once.
- Join this table with the variable values using `data.table` for fast aggregation.
- Compute max, min, mean in a single grouped operation per variable.
- Avoid per-row `lapply` and repeated indexing.

This approach reduces complexity from millions of nested R loops to efficient C-backed aggregation.

---

**Working R Code:**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Precompute neighbor lookup in long format
build_neighbor_table <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  row_ids <- seq_len(nrow(data))
  res_list <- vector("list", length(row_ids))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    neighbor_idx <- idx_lookup[neighbor_keys]
    res_list[[i]] <- data.table(
      focal_idx = i,
      neighbor_idx = as.integer(neighbor_idx[!is.na(neighbor_idx)])
    )
  }
  
  rbindlist(res_list)
}

neighbor_table <- build_neighbor_table(cell_data, id_order, rook_neighbors_unique)

# Join with cell_data for neighbor values
compute_neighbor_features_dt <- function(data, neighbor_table, var_name) {
  vals <- data[, .(idx = .I, val = get(var_name))]
  nt <- merge(neighbor_table, vals, by.x = "neighbor_idx", by.y = "idx", all.x = TRUE)
  
  # Aggregate neighbor stats
  stats <- nt[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = focal_idx]
  
  # Merge back to original data
  data[stats$focal_idx, paste0(var_name, "_nbr_max") := stats$max_val]
  data[stats$focal_idx, paste0(var_name, "_nbr_min") := stats$min_val]
  data[stats$focal_idx, paste0(var_name, "_nbr_mean") := stats$mean_val]
}

# Apply for all neighbor source variables
for (var_name in neighbor_source_vars) {
  compute_neighbor_features_dt(cell_data, neighbor_table, var_name)
}
```

---

**Why This Works:**  
- Eliminates repeated `lapply` over millions of rows.
- Uses `data.table` joins and group-by for efficient aggregation.
- Preserves original estimands (max, min, mean).
- Scales to millions of rows within reasonable time and memory on a 16 GB laptop.

**Expected Performance Gain:**  
From 86+ hours to a few hours or less, depending on disk I/O and RAM.
END_RESPONSE>>>
