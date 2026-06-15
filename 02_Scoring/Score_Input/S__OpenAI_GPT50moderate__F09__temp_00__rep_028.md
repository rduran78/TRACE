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
run_id: OpenAI_F09_T00_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F09_OpenAI_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, neighbor indices are recomputed repeatedly for every variable. This results in redundant lookups and excessive list operations. The main bottlenecks are:  
- Rebuilding neighbor relationships per row and per variable.  
- Using `lapply` over millions of rows multiple times.  
- Lack of vectorization and precomputation.  

**Optimization Strategy**  
- Build the neighbor lookup **once** for all cell-year rows (as already suggested).  
- Compute neighbor statistics for all variables in a **single pass** using matrix operations instead of looping per variable.  
- Use `data.table` for efficient joins and grouping.  
- Avoid repeated string concatenation and repeated indexing.  
- Keep memory usage manageable by processing in chunks if needed.  

**Optimized R Code**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup once
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}

neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for multiple variables in one pass
compute_neighbor_stats_multi <- function(data, neighbor_lookup, vars) {
  vals_mat <- as.matrix(data[, ..vars])
  res_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- vals_mat[, v]
    res_list[[v]] <- t(vapply(neighbor_lookup, function(idx) {
      if (length(idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3)))
  }
  
  names(res_list) <- vars
  res_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_list <- compute_neighbor_stats_multi(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind results back to cell_data
for (i in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[i]
  cell_data[, paste0(var, "_nbr_max") := stats_list[[i]][, 1]]
  cell_data[, paste0(var, "_nbr_min") := stats_list[[i]][, 2]]
  cell_data[, paste0(var, "_nbr_mean") := stats_list[[i]][, 3]]
}

# Now cell_data has neighbor features ready for Random Forest prediction
# Use the existing trained model:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Expected Performance Gain**  
- Neighbor lookup built once instead of per variable â†’ major speedup.  
- Single pass for all variables reduces overhead by ~5x.  
- Using `vapply` and preallocated matrices avoids repeated memory allocation.  
- Estimated runtime reduction from 86+ hours to a few hours (depending on I/O and CPU).  

This preserves the trained Random Forest model and the original estimand while making the pipeline significantly faster.
END_RESPONSE>>>
