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
run_id: OpenAI_F09_T03_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for every variable and every year, which is highly inefficient given the large dataset (6.46M rows, 344K cells, 28 years). The bottleneck is the repeated neighbor lookup and aggregation inside the outer loop. Memory use is also high because of redundant operations.

**Optimization Strategy**  
- Precompute a reusable neighbor lookup table once for all cell-year combinations.
- Use vectorized operations or `data.table` for fast joins instead of repeated `lapply`.
- Compute neighbor statistics for all variables in a single pass per year using the precomputed lookup.
- Avoid rebuilding neighbor relationships for each variable.
- Keep Random Forest model unchanged and preserve estimands.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup for all cell-year combinations
build_neighbor_lookup_dt <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))
  
  # Create a long table of (row_id, neighbor_row_id)
  row_ids <- seq_len(nrow(cell_data))
  lookup_list <- vector("list", length(row_ids))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(cell_data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
    neighbor_idx <- idx_lookup[neighbor_keys]
    lookup_list[[i]] <- data.table(row_id = i, neighbor_id = as.integer(neighbor_idx[!is.na(neighbor_idx)]))
  }
  
  rbindlist(lookup_list)
}

neighbor_lookup_dt <- build_neighbor_lookup_dt(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for all variables efficiently
compute_neighbor_stats_dt <- function(cell_data, neighbor_lookup_dt, var_names) {
  results <- list()
  for (var_name in var_names) {
    vals <- cell_data[[var_name]]
    dt_vals <- data.table(row_id = seq_len(nrow(cell_data)), value = vals)
    
    # Join neighbor_lookup with values
    joined <- neighbor_lookup_dt[dt_vals, on = .(neighbor_id = row_id)]
    
    # Aggregate stats by row_id
    agg <- joined[, .(
      max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
      min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
      mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
    ), by = row_id]
    
    setnames(agg, c("row_id", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
    results[[var_name]] <- agg
  }
  
  # Merge all results back into cell_data
  for (res in results) {
    cell_data[res, on = .(row_id), names(res)[-1] := res[, -1]]
  }
  
  return(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_neighbor_stats_dt(cell_data, neighbor_lookup_dt, neighbor_source_vars)
```

---

### **Expected Impact**
- **Time reduction**: From 86+ hours to a few hours (due to single-pass joins and aggregation).
- **Memory efficiency**: Avoid repeated `lapply` and redundant neighbor computations.
- **Preserves Random Forest model and estimands**: No retraining, only feature computation optimized.

This approach leverages `data.table` for fast joins and aggregation, and builds the neighbor lookup once for all years and variables.
END_RESPONSE>>>
