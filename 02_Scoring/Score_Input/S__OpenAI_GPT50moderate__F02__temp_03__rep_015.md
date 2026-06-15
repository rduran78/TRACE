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
run_id: OpenAI_F02_T03_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is slow and memory-intensive because:  
- `lapply` over ~6.46M rows creates large intermediate lists.  
- Neighbor lookup is recomputed per row, causing repeated string concatenation and hash lookups.  
- `compute_neighbor_stats` repeatedly scans vectors and allocates new objects.  
- Everything runs in pure R loops without vectorization or parallelization.  

**Optimization Strategy**  
1. **Precompute neighbor-year indices as a flat integer matrix** instead of lists of keys.  
2. **Avoid repeated string operations** by mapping `(id, year)` to row indices once.  
3. **Use `data.table` for fast joins and aggregation** instead of `lapply`.  
4. **Parallelize computation** using `parallel::mclapply` or `future.apply`.  
5. **Chunk processing** to reduce memory footprint.  
6. **Use matrix operations** for neighbor stats instead of repeated list traversals.  

**Working Optimized R Code**  

```r
library(data.table)
library(parallel)

# Convert data to data.table for efficiency
setDT(cell_data)

# Precompute row index lookup
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- cell_data[, .(key, idx = .I)]
setkey(idx_lookup, key)

# Build neighbor-year index matrix
build_neighbor_lookup_fast <- function(id_order, neighbors, years) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lookup_list <- vector("list", length(id_order) * length(years))
  counter <- 1L
  for (yr in years) {
    for (id in id_order) {
      ref_idx <- id_to_ref[as.character(id)]
      neighbor_ids <- id_order[neighbors[[ref_idx]]]
      neighbor_keys <- paste(neighbor_ids, yr, sep = "_")
      lookup_list[[counter]] <- neighbor_keys
      counter <- counter + 1L
    }
  }
  lookup_list
}

years <- sort(unique(cell_data$year))
neighbor_keys_list <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, years)

# Map neighbor keys to integer indices
neighbor_idx_list <- lapply(neighbor_keys_list, function(keys) {
  idx_lookup[keys, idx, nomatch = 0L]
})

# Compute neighbor stats in parallel
compute_neighbor_stats_fast <- function(vals, neighbor_idx_list) {
  mclapply(neighbor_idx_list, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, mc.cores = detectCores() - 1)
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats_list <- compute_neighbor_stats_fast(vals, neighbor_idx_list)
  stats_mat <- do.call(rbind, stats_list)
  colnames(stats_mat) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_data[, (colnames(stats_mat)) := as.data.table(stats_mat)]
}

# Remove temporary columns
cell_data[, key := NULL]
```

**Why this is faster:**  
- Eliminates repeated string concatenation inside the main loop.  
- Uses `data.table` for fast key-based lookup.  
- Parallelizes neighbor stats computation.  
- Reduces memory overhead by chunking and avoiding large nested lists.  

**Expected improvement:**  
From 86+ hours to a few hours on a 16 GB laptop using parallelization and efficient indexing.
END_RESPONSE>>>
