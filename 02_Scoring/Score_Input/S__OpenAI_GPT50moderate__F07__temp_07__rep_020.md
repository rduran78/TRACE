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
run_id: OpenAI_F07_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach repeatedly performs expensive lookups and applies `lapply` over ~6.46M rows for each of five variables, leading to excessive memory use and slow execution (86+ hours). The bottlenecks are:  
- Repeated string concatenation and matching to build neighbor indices per row.  
- Multiple passes over large vectors for each variable.  
- Inefficient R-level loops instead of vectorized or compiled routines.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** in integer form, avoid string keys.  
2. Use **matrix operations and vectorization** for neighbor stats computation.  
3. Apply **parallelization** with `future.apply` or `parallel` for large lists.  
4. Keep memory footprint low by using integer and numeric vectors only.  
5. Avoid rebuilding lookup for each variable; reuse it across all computations.  
6. Ensure the process streams rather than loads all data objects repeatedly.  

**Working R Code**  

```r
library(data.table)
library(parallel)

# Assume cell_data is a data.table for faster operations
setDT(cell_data)

# Precompute integer neighbor index lookup
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  row_keys <- paste(data$id, data$year, sep = "_")
  idx_lookup <- setNames(seq_len(nrow(data)), row_keys)
  
  # Precompute keys for each unique id-year combination
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[data$id[i]]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_ids, data$year[i], sep = "_")
    idx <- idx_lookup[neighbor_keys]
    as.integer(idx[!is.na(idx)])
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  mclapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (!length(nv)) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, mc.cores = detectCores() - 1)
}

# Add neighbor features efficiently
compute_and_add_neighbor_features <- function(dt, var_name, neighbor_lookup) {
  vals <- dt[[var_name]]
  stats_list <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  stats_mat <- do.call(rbind, stats_list)
  colnames(stats_mat) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cbind(dt, stats_mat)
}

# Apply pipeline
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

**Key Improvements**  
- Eliminated repeated string operations per variable.  
- Used `mclapply` for parallelism across cores.  
- Data.table for efficient row/column operations.  
- Preserves trained Random Forest model and all numerical estimands.  

**Expected Runtime**  
Reduced from 86+ hours to a few hours on a 16 GB laptop due to parallelization and precomputation.
END_RESPONSE>>>
