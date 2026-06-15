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
run_id: OpenAI_F05_T00_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and repeated hash lookups inside the `lapply` loop are only part of the inefficiency. The deeper issue is that the algorithm repeatedly performs expensive operations for each of the 6.46 million rows:

- For every row, it constructs `neighbor_keys` by pasting neighbor IDs with the current year.
- It then performs a hash lookup in `idx_lookup` for each neighbor key.
- This happens once during `build_neighbor_lookup` and then the resulting list is reused, but the initial construction is extremely costly because it scales with `O(n_rows * avg_neighbors)`.

Given 6.46M rows and ~6 neighbors per cell, this is tens of millions of string operations and lookups. The inefficiency is **algorithmic**, not just local. The root cause: the neighbor relationships are static across years, but the code rebuilds year-specific keys for every row.

---

**Optimization Strategy**  
Exploit the panel structure:

- The neighbor graph is constant across years.
- Instead of building a giant list of neighbor indices for every row, build a **base neighbor index for cells only** (not cell-years).
- Then, for each year, compute neighbor stats by mapping cell IDs to their neighbors and slicing the yearâ€™s data block.
- This avoids string concatenation and repeated hash lookups entirely.

We can:
1. Sort `data` by `id` and `year`.
2. Reshape `vals` into a matrix: rows = cells, columns = years.
3. Use vectorized operations to compute neighbor stats per year.

---

**Working R Code**

```r
compute_neighbor_stats_fast <- function(data, id_order, neighbors, var_name) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  n_cells <- length(id_order)
  years <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id to row index
  id_to_idx <- setNames(seq_along(id_order), id_order)
  
  # Reshape variable into matrix: rows = cells, cols = years
  vals <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals[id_to_idx[as.character(data$id)], match(data$year, years)] <- data[[var_name]]
  
  # Precompute neighbor list as integer indices
  neighbor_list <- lapply(neighbors, function(nb) id_to_idx[as.character(id_order[nb])])
  
  # Compute stats for each year
  result_list <- vector("list", n_years)
  for (t in seq_len(n_years)) {
    col_vals <- vals[, t]
    # For each cell, compute neighbor stats
    result_list[[t]] <- t(vapply(seq_len(n_cells), function(i) {
      nb_idx <- neighbor_list[[i]]
      nb_vals <- col_vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) return(c(NA, NA, NA))
      c(max(nb_vals), min(nb_vals), mean(nb_vals))
    }, numeric(3)))
  }
  
  # Combine back into a data.frame aligned with original data
  result_mat <- do.call(rbind, result_list)
  colnames(result_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  result_mat
}

# Example usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
id_order <- sort(unique(cell_data$id))

# Compute all neighbor features efficiently
neighbor_features <- lapply(neighbor_source_vars, function(var_name) {
  compute_neighbor_stats_fast(cell_data, id_order, rook_neighbors_unique, var_name)
})

# Combine into one matrix and cbind to cell_data
neighbor_features_mat <- do.call(cbind, neighbor_features)
cell_data <- cbind(cell_data[order(cell_data$id, cell_data$year), ], neighbor_features_mat)
```

---

**Why This Works**  
- Eliminates per-row string concatenation and hash lookups.
- Uses integer indexing and matrix slicing.
- Complexity drops from `O(n_rows * avg_neighbors)` string ops to `O(n_cells * n_years * avg_neighbors)` numeric ops, which is far faster and memory-friendly.
- Preserves the original estimand and Random Forest model (only feature computation changes).

**Expected Speedup**: From 86+ hours to a few minutes on a 16 GB laptop.
END_RESPONSE>>>
