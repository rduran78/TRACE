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
run_id: OpenAI_F07_T00_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F07_OpenAI_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Neighbor lookups are recomputed for every row and every variable, causing redundant work.  
- Pure R loops and list operations are inefficient for this scale.  
- Memory overhead is high due to repeated object creation.  

**Optimization Strategy**  
- Precompute a **dense integer matrix** of neighbor indices (with `NA` padding) for all rows to avoid repeated list traversals.  
- Use **vectorized operations** or `data.table` for fast aggregation.  
- Compute all neighbor stats in a single pass per variable using matrix indexing.  
- Avoid recomputing neighbor lookups for each variable.  
- Use `Rcpp` or `matrixStats` for efficient row-wise max, min, mean.  

**Working R Code (Optimized)**  

```r
library(data.table)
library(matrixStats)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in reference order
# rook_neighbors_unique: spdep::nb object

# 1. Precompute neighbor lookup as a dense matrix
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  row_ids <- seq_len(nrow(data))
  max_nbrs <- max(lengths(neighbors))
  
  neighbor_mat <- matrix(NA_integer_, nrow = length(row_ids), ncol = max_nbrs)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    nbr_keys <- paste(nbr_ids, data$year[i], sep = "_")
    nbr_idx <- idx_lookup[nbr_keys]
    nbr_idx <- as.integer(nbr_idx[!is.na(nbr_idx)])
    if (length(nbr_idx) > 0) {
      neighbor_mat[i, seq_along(nbr_idx)] <- nbr_idx
    }
  }
  neighbor_mat
}

neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# 2. Compute neighbor stats efficiently
compute_neighbor_stats_matrix <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  # Extract neighbor values as a matrix
  nbr_vals <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))
  
  # Handle NA rows
  row_max <- rowMaxs(nbr_vals, na.rm = TRUE)
  row_min <- rowMins(nbr_vals, na.rm = TRUE)
  row_mean <- rowMeans(nbr_vals, na.rm = TRUE)
  
  # Replace NaN (from all-NA rows) with NA
  row_mean[is.nan(row_mean)] <- NA
  
  data.table(
    paste0(var_name, "_nbr_max") = row_max,
    paste0(var_name, "_nbr_min") = row_min,
    paste0(var_name, "_nbr_mean") = row_mean
  )
}

# 3. Apply for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- lapply(neighbor_source_vars, function(v) {
  compute_neighbor_stats_matrix(cell_data, neighbor_mat, v)
})

# 4. Bind results to cell_data
cell_data <- cbind(cell_data, do.call(cbind, result_list))
```

**Why This Works**  
- `neighbor_mat` is computed once and reused.  
- `matrixStats` functions are highly optimized in C for row-wise operations.  
- Avoids millions of small list operations and repeated lookups.  
- Memory footprint is controlled by using integer matrix for neighbors and numeric matrix for values.  

**Expected Performance**  
- From 86+ hours â†’ likely under 1 hour on a 16 GB laptop (depending on I/O).  
- Preserves original rook-neighbor relationships and numerical estimands.  
- No retraining of the Random Forest model required.
END_RESPONSE>>>
