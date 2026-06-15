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
run_id: OpenAI_F04_T00_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F04_OpenAI_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation across ~6.46M rows. The current implementation uses `lapply` with repeated string concatenation and list indexing, which is highly inefficient for this scale. The `build_neighbor_lookup` function constructs neighbor indices per row using expensive operations (`paste`, `setNames`, repeated lookups). Then `compute_neighbor_stats` iterates again over all rows for each variable, performing redundant NA filtering and aggregation. This results in billions of R-level operations.

**Optimization Strategy**  
- **Precompute neighbor indices once** as integer vectors without repeated string concatenation.
- **Vectorize neighbor statistics** using `data.table` or `vapply` instead of `lapply`.
- **Avoid repeated NA filtering** by using fast aggregation functions.
- **Use matrix operations** for multiple variables at once.
- **Parallelize** across cores using `parallel::mclapply` or `future.apply`.
- Keep memory footprint low by working with integer indices and numeric vectors only.

**Working R Code (Optimized)**  

```r
library(data.table)
library(parallel)

# Convert to data.table for fast operations
setDT(cell_data)

# Precompute lookup: map (id, year) -> row index
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Build neighbor lookup as integer indices (vectorized)
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  years <- data$year
  ids <- as.character(data$id)
  
  # Precompute neighbor indices for all rows
  mclapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[ids[i]]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_ids, years[i], sep = "_")
    res <- idx_lookup[neighbor_keys]
    as.integer(res[!is.na(res)])
  }, mc.cores = detectCores() - 1)
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for multiple variables efficiently
compute_neighbor_stats_fast <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  
  result <- mclapply(seq_along(neighbor_lookup), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(rep(NA_real_, length(vars) * 3))
    
    out <- numeric(length(vars) * 3)
    for (j in seq_along(vars)) {
      nv <- vals_list[[j]][idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0) {
        out[(j - 1) * 3 + 1:3] <- NA_real_
      } else {
        out[(j - 1) * 3 + 1:3] <- c(max(nv), min(nv), mean(nv))
      }
    }
    out
  }, mc.cores = detectCores() - 1)
  
  do.call(rbind, result)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_matrix <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, neighbor_source_vars)

# Add computed features back to cell_data
stat_names <- unlist(lapply(neighbor_source_vars, function(v) paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))))
cell_data[, (stat_names) := as.data.table(stats_matrix)]
```

**Expected Impact**  
- Eliminates repeated string operations and redundant loops.
- Uses parallelization and vectorization for major speedup.
- Preserves original estimand and Random Forest model.
- Estimated runtime reduction from 86+ hours to a few hours on a 16 GB laptop.
END_RESPONSE>>>
