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
run_id: OpenAI_F04_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is not the Random Forest inference but the neighbor-feature construction, specifically the `lapply`-based row-wise operations in `build_neighbor_lookup` and `compute_neighbor_stats`. For 6.46M rows Ã— 5 variables, repeated `lapply` calls and string concatenations lead to excessive overhead. The operations are memory- and CPU-bound due to millions of lookups in R lists.

---

**Optimization Strategy**  
1. **Avoid per-row string concatenations**: Precompute key mappings as integer indices, not strings.  
2. **Vectorize neighbor computations**: Collapse repeated `lapply` calls to faster structures (e.g., `data.table`, matrix operations).  
3. **Precompute neighbor indices once**: Store cell-year indices in a flat integer matrix or list for reuse.  
4. **Use efficient packages**: `data.table` for grouping, `matrixStats` for summary stats.  
5. **Parallelization**: Use `future.apply` or `parallel` on multi-core CPU for computing neighbor summaries.  

The goal: Build neighbor index lookup once, then compute stats using vectorized operations on integer vectors instead of string manipulations.

---

**Working R Code (Optimized Approach)**  

```r
library(data.table)
library(matrixStats)
library(future.apply)

# Convert to data.table for speed
setDT(cell_data)

# Precompute index lookup (integer-based)
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Build neighbor indices efficiently
build_neighbor_indices <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  row_ids <- seq_len(nrow(data))
  
  data[, {
    ref_idx <- id_to_ref[as.character(id)]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, year, sep = "_")
    .(neighbor_idx = list(as.integer(idx_lookup[neighbor_keys])))
  }, by = seq_len(nrow(data))]$neighbor_idx
}

neighbor_lookup <- build_neighbor_indices(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats in parallel
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  future_lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  })
}

# Apply for each variable and bind to cell_data
plan(multisession, workers = parallel::detectCores() - 1)

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats_list <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  
  # Combine results quickly
  stats_matrix <- do.call(rbind, stats_list)
  new_cols <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data[, (new_cols) := as.data.table(stats_matrix)]
}

# Remove helper columns
cell_data[, key := NULL]
```

---

**Key Changes & Performance Gains**  
- Replaced repeated string-based matching with integer lookups.
- Batched computations using `future_lapply` for parallel execution.
- Leveraged `data.table` for efficient joins and column addition.
- Reduced runtime drastically (from 86+ hours to a few hours or less on multi-core machine).

This preserves the estimand (same neighbor-derived statistics), avoids retraining the Random Forest, and uses existing machine constraints efficiently.
END_RESPONSE>>>
