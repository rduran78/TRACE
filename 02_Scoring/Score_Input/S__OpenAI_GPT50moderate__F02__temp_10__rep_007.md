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
run_id: OpenAI_F02_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current process is extremely memory and compute intensive because `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M rows, creating large lists with repeated string operations (`paste` lookups for every row) and multiple allocations. These operations scale poorly with such a large dataset. Most time and memory overhead comes from:  
1. Repeated character concatenation for keys.  
2. Large `lapply` constructing millions of small vectors.  
3. Lack of vectorization and parallelization.  

**Optimization Strategy:**  
- **Avoid repeated string concatenation**: Precompute `(id, year)` pairs and map them with integer indexing rather than text keys.  
- **Flatten neighbor lookup to matrix form** where rows align with data rows (or keep adjacency in integers).  
- **Parallelize computation** using `data.table` or `future` + `furrr`.  
- **Process in chunks** rather than all rows at once to fit laptop memory.  
- Replace `lapply` with `vapply` or vectorized aggregations from `data.table`.  
- Store `neighbor_lookup` as `list(integer)` indexed by row number directly without paste operations.  

**Working Example with `data.table` and chunking:**  

```r
library(data.table)
library(parallel)

# Assume cell_data is a data.frame; convert to data.table for speed
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Precompute index mapping for fast join
cell_data[, row_idx := .I]

# Build neighbor lookup (integer-based, chunked)
build_neighbor_lookup_fast <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  n <- nrow(cell_data)
  lookup <- vector("list", n)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[[as.character(cell_data$id[i])]]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    # Match neighbors for same year:
    idx <- cell_data[.(neighbor_ids, cell_data$year[i]), on = .(id, year), row_idx]
    lookup[[i]] <- idx[!is.na(idx)]
  }
  lookup
}

# Compute stats in parallel
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  cl <- makeCluster(detectCores() - 1)
  on.exit(stopCluster(cl))
  
  clusterExport(cl, c("vals", "neighbor_lookup"), envir = environment())
  
  res <- parLapply(cl, neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    v <- vals[idx]
    v <- v[!is.na(v)]
    if (length(v) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(v), min(v), mean(v))
  })
  do.call(rbind, res)
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_lookup)
  cell_data[, paste0(var_name, "_nbr_max") := stats[, 1]]
  cell_data[, paste0(var_name, "_nbr_min") := stats[, 2]]
  cell_data[, paste0(var_name, "_nbr_mean") := stats[, 3]]
}
```

**Why this helps:**  
- Eliminates costly string operations by using integer indexing.  
- Processes with efficient `data.table` joins rather than `paste` keys.  
- Uses parallel processing for neighbor stats (significant speedup on multi-core laptops).  
- Memory footprint is reduced because we avoid creating 6M long strings.  

**Expected Outcome:**  
Instead of 86+ hours, depending on cores and chunk size, the process should complete in a few hours on a 16GB laptop without exhausting memory and without changing the numerical estimand or touching the Random Forest model.
END_RESPONSE>>>
