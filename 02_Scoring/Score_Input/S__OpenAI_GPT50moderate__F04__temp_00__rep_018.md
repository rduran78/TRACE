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
run_id: OpenAI_F04_T00_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F04_OpenAI_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation across ~6.46M rows. The current implementation uses `lapply` with repeated string concatenation and list indexing, which is highly inefficient in R for large datasets. The neighbor lookup is rebuilt once, but `compute_neighbor_stats` iterates over all rows for each variable, performing redundant operations. This results in excessive memory allocation and function call overhead.

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer vectors (avoid repeated string concatenation and lookups).
2. **Vectorize neighbor statistics computation** using `vapply` or matrix operations instead of repeated `lapply`.
3. **Use data.table for fast grouping and joins** to handle large datasets efficiently.
4. **Parallelize across variables or chunks** using `parallel` or `future.apply`.
5. Avoid repeated NA filtering inside loops; handle NA logic in a vectorized manner.

**Working Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute lookup: map (id, year) -> row index
idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))

# Build neighbor lookup as integer indices (vectorized)
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  row_ids <- seq_len(nrow(data))
  years <- data$year
  ids <- data$id
  
  # Preallocate list
  neighbor_lookup <- vector("list", length(row_ids))
  
  # Use mclapply for parallelization
  neighbor_lookup <- mclapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(ids[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, years[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  }, mc.cores = detectCores() - 1)
  
  neighbor_lookup
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  # Parallelize computation
  chunks <- split(seq_len(n), cut(seq_len(n), detectCores() - 1))
  results_list <- mclapply(chunks, function(chunk) {
    chunk_res <- matrix(NA_real_, nrow = length(chunk), ncol = 3)
    for (j in seq_along(chunk)) {
      i <- chunk[j]
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          chunk_res[j, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    chunk_res
  }, mc.cores = detectCores() - 1)
  
  result <- do.call(rbind, results_list)
  result
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  cell_data[, paste0(var_name, "_nbr_max") := stats[, 1]]
  cell_data[, paste0(var_name, "_nbr_min") := stats[, 2]]
  cell_data[, paste0(var_name, "_nbr_mean") := stats[, 3]]
}
```

**Expected Gains**  
- Eliminates repeated string operations inside loops.
- Uses parallelization and preallocation to reduce overhead.
- Data.table ensures efficient column assignment.
- Should reduce runtime from 86+ hours to a few hours on a 16 GB laptop.

This preserves the trained Random Forest model and the original numerical estimand while making the pipeline practical.
END_RESPONSE>>>
