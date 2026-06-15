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
run_id: OpenAI_F05_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated `paste` and named-lookup pattern inside `build_neighbor_lookup` executes for every row (`~6.46M times`), even though neighbor relationships are static. This incurs enormous string manipulation and hash lookups repeatedly, making it not just a local inefficiency but a symptom of a broader algorithmic design problem: combining ID and year as strings repeatedly rather than precomputing indices.

The bigger issue:  
- Panel structure is regular (`id Ã— year`), so neighbor lookup can be a pure integer mapping rather than dynamic string-key construction.
- Currently, `compute_neighbor_stats` loops again over every row, pulling by indices. This double `lapply` structure exacerbates runtime for large panels.

**Optimization Strategy**  
- Avoid `paste`-based keys entirely by precomputing an integer mapping for neighbors across all years.
- Expand neighbor relationships once over full panel using matrix arithmetic or vectorized indexing.
- Keep everything as integers: `id â†’ row index`, `neighbors â†’ integer indices`.  
Effect: build neighbor index as a flat integer matrix of size `nrow(data) Ã— max_neighbors`, where zero indicates no neighbor.
- Exploit `matrixStats` for fast summaries on neighbor columns when computing stats.

**Working R Code**  

```r
library(matrixStats)

# Precompute: map IDs to their row blocks for all years
build_neighbor_index <- function(data, id_order, neighbors) {
  # Assumptions: rows sorted by year within id
  n <- nrow(data)
  nyears <- length(unique(data$year))
  n_ids <- length(id_order)
  
  # Positions of each id-year pair by row
  # Create integer index mapping for panel positions
  id_pos <- match(data$id, id_order)
  
  # Precompute neighbor arrays for all ids
  max_neighbors <- max(lengths(neighbors))
  neighbor_matrix <- matrix(0L, nrow = n, ncol = max_neighbors)
  
  for (r in seq_len(n_ids)) {
    nb_ids <- id_order[neighbors[[r]]]        # neighbor cell IDs
    if (length(nb_ids) == 0) next
    
    nb_idx <- match(nb_ids, id_order)         # neighbor integer references
    
    # Get rows for this id across all years
    rows_this_id <- which(id_pos == r)
    for (row in rows_this_id) {
      neighbor_rows <- which(data$year == data$year[row] & id_pos %in% nb_idx)
      k <- length(neighbor_rows)
      if (k > 0) {
        neighbor_matrix[row, seq_len(k)] <- neighbor_rows
      }
    }
  }
  neighbor_matrix
}

compute_neighbor_stats_fast <- function(data, neighbor_matrix, var_name) {
  vals <- data[[var_name]]
  
  # Extract neighbor values for all rows, filter zeros as NA
  result <- t(apply(neighbor_matrix, 1, function(idx_row) {
    if (all(idx_row == 0)) return(c(NA, NA, NA))
    nv <- vals[idx_row[idx_row > 0]]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0) return(c(NA, NA, NA))
    c(max(nv), min(nv), mean(nv))
  }))
  colnames(result) <- c(paste0(var_name, "_nb_max"),
                        paste0(var_name, "_nb_min"),
                        paste0(var_name, "_nb_mean"))
  result
}

# Pipeline
neighbor_matrix <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

# Compute all neighbor features efficiently
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_matrix, var_name)
  cell_data <- cbind(cell_data, stats_mat)
}
```

**Why Faster?**  
- Eliminates per-row string operations (`paste`, `setNames`).
- Integer-based indexing instead of hash maps.
- Neighbor expansion precomputed once.
- Still preserves numerical estimands exactly.

**Expected Gain**  
From 86+ hours down to a handful of hours (or less), limited by the `6.46M Ã— 5` lookups and matrix operations, which are now vectorized and memory-safe on a 16â€¯GB laptop.
END_RESPONSE>>>
