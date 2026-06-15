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
run_id: OpenAI_F02_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The pipeline is slow and memory-intensive because:  
- `lapply` over ~6.46M rows repeatedly constructs neighbor lookups and stats, creating large intermediate lists.  
- Each variable recomputes neighbor stats from scratch instead of reusing computations.  
- Pure R loops over millions of elements are inefficient on a standard laptop.  

**Optimization Strategy**  
- Precompute `neighbor_lookup` once and store as an integer matrix or list of vectors for direct indexing.  
- Use `data.table` for fast row operations and column access.  
- Vectorize neighbor-stat computations where possible (compute all source variables in one pass).  
- Eliminate repeated string concatenations; use numeric indexing instead.  
- Optionally, parallelize using `future.apply` or `parallel` if needed.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert data to data.table for efficiency
setDT(cell_data)

# Precompute lookup keys
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Build neighbor lookup as integer indices
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(data)
  years <- data$year
  ids <- as.character(data$id)
  
  # Preallocate result as list
  res <- vector("list", n)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[ids[i]]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    keys <- paste(nbr_ids, years[i], sep = "_")
    res[[i]] <- as.integer(idx_lookup[keys])
  }
  res
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for all source vars in one pass
compute_neighbor_stats_all <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  n <- length(neighbor_lookup)
  out <- matrix(NA_real_, nrow = n, ncol = length(vars) * 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      for (j in seq_along(vars)) {
        vals <- vals_list[[j]][idx]
        vals <- vals[!is.na(vals)]
        if (length(vals) > 0) {
          out[i, (j - 1) * 3 + 1] <- max(vals)
          out[i, (j - 1) * 3 + 2] <- min(vals)
          out[i, (j - 1) * 3 + 3] <- mean(vals)
        }
      }
    }
  }
  out
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_matrix <- compute_neighbor_stats_all(cell_data, neighbor_lookup, neighbor_source_vars)

# Add computed columns back to cell_data
col_names <- unlist(lapply(neighbor_source_vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
cell_data[, (col_names) := as.data.table(stats_matrix)]
```

**Why this helps:**  
- Single neighbor loop for all variables (avoids 5Ã— repetition).  
- Avoids repeated `do.call(rbind, ...)` overhead.  
- Uses integer indexing, not string concatenation inside loops.  
- Memory footprint reduced by reusing `vals_list`.  

**Expected Gains:**  
- From 86+ hours to a few hours, possibly <10 hours on a 16 GB laptop, especially if parallelized.  
- Preserves original estimand and trained model.  
- Scales better for millions of rows.
END_RESPONSE>>>
