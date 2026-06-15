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
run_id: OpenAI_F04_T03_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation across ~6.46M rows and 5 variables using `lapply` and repeated string concatenation (`paste`) inside `build_neighbor_lookup`. This creates massive overhead in Râ€™s interpreter and memory allocations. The neighbor lookup is recomputed per row, and then statistics are computed with repeated list traversals.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer vectors (avoid repeated string operations).  
2. **Vectorize neighbor statistics computation** using `vapply` or matrix operations instead of `lapply`.  
3. Use **data.table** for fast joins and column access.  
4. Avoid repeated NA filtering inside loops; pre-mask NA values.  
5. Parallelize across variables using `future.apply` or `parallel`.  

**Working R Code**  

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute lookup: map (id -> row indices by year)
idx_lookup <- cell_data[, .I, by = .(id, year)]
setkey(idx_lookup, id, year)

# Build neighbor lookup as integer indices without paste
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  n <- nrow(data)
  result <- vector("list", n)
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[data$id[i]]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    # Direct join for year
    neighbor_rows <- idx_lookup[J(neighbor_ids, data$year[i]), I]
    result[[i]] <- neighbor_rows
  }
  result
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(res) <- c("max", "min", "mean")
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

# Parallelize across variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("cell_data", "neighbor_lookup", "compute_neighbor_stats_fast"))

neighbor_features <- parLapply(cl, neighbor_source_vars, function(var_name) {
  vals <- cell_data[[var_name]]
  compute_neighbor_stats_fast(vals, neighbor_lookup)
})
stopCluster(cl)

# Bind results
for (i in seq_along(neighbor_source_vars)) {
  nm <- neighbor_source_vars[i]
  stats <- neighbor_features[[i]]
  cell_data[, paste0(nm, "_nbr_max") := stats[, 1]]
  cell_data[, paste0(nm, "_nbr_min") := stats[, 2]]
  cell_data[, paste0(nm, "_nbr_mean") := stats[, 3]]
}
```

**Expected Impact**  
- Eliminates repeated string concatenation and redundant lookups.  
- Reduces overhead by using integer joins and matrix assignments.  
- Parallelization across 5 variables on 16 GB machine should cut runtime from 86+ hours to a few hours.  
- Preserves original estimand and trained Random Forest model.
END_RESPONSE>>>
