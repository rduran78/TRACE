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
run_id: OpenAI_F05_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The repeated string-key construction (`paste(data$id, data$year, sep = "_")`) and subsequent lookup inside `lapply` is not just a local inefficiencyâ€”it indicates a broader algorithmic bottleneck. For every row (â‰ˆ6.46M), you perform multiple string concatenations and dictionary lookups for neighbors, which scales very poorly. This is essentially an `O(N * avg_neighbors)` operation with expensive character operations inside. The repeated computation of `neighbor_keys` and indexing into `idx_lookup` dominates runtime.

**Optimization Strategy:**  
Eliminate string-based keys and repeated lookups inside the inner loop. Instead:  
1. Precompute all mappings using numeric indices upfront.  
2. Represent neighbor relationships in cell-year space as integer indices, avoiding repeated string concatenations.  
3. Use vectorized or matrix-based operations where feasible.  

The idea: expand neighbor relationships across years once, then reuse them. This converts nested loops and repeated concatenation into a single efficient integer-based lookup table.

---

### **Optimized Approach**
- Precompute a numeric mapping from `id` to row indices by year.
- Construct a global neighbor index matrix for all rows and store it once.
- Use this matrix to compute neighbor stats without recomputing keys.

---

### **Working R Code**
```r
# Precompute mapping of (id, year) -> row index
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n_rows <- nrow(data)
  n_years <- length(unique(data$year))
  max_neighbors <- max(sapply(neighbors, length))

  # Map id to position in id_order
  id_to_ref <- match(data$id, id_order)

  # Initialize matrix: rows = n_rows, cols = max_neighbors
  # Fill with NA for missing neighbors
  neighbor_matrix <- matrix(NA_integer_, nrow = n_rows, ncol = max_neighbors)

  # Precompute year groups for efficient access
  year_vec <- data$year

  # For each row, fill neighbor indices
  for (i in seq_len(n_rows)) {
    ref_idx <- id_to_ref[i]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neighbor_ids) > 0) {
      # Find rows where id matches neighbor_ids and year == current year
      # Instead of string keys, use a fast lookup via logical indexing
      current_year <- year_vec[i]
      neighbor_rows <- which(data$id %in% neighbor_ids & year_vec == current_year)
      # Fill in matrix positions
      neighbor_matrix[i, seq_along(neighbor_rows)] <- neighbor_rows
    }
  }
  neighbor_matrix
}

compute_neighbor_stats_fast <- function(data, neighbor_matrix, var_name) {
  vals <- data[[var_name]]
  n_rows <- nrow(data)
  result <- matrix(NA_real_, nrow = n_rows, ncol = 3)  # max, min, mean

  for (i in seq_len(n_rows)) {
    idx <- neighbor_matrix[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Usage:
neighbor_matrix <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_matrix, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Performance Gains**
- Eliminates repeated string concatenation and dictionary lookups.
- Converts neighbor relationships to numeric indexing.
- Memory footprint: `neighbor_matrix` is large but manageable (~6.46M Ã— max_neighbors integers).
- Runtime: **orders of magnitude faster**, likely reducing from 86+ hours to a few hours on 16GB RAM laptop.

**Numerical Estimand:** Preserved, as the logic for identifying neighbors and computing max/min/mean remains unchanged.
END_RESPONSE>>>
