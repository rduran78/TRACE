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
run_id: OpenAI_F05_T00_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and repeated hash lookups inside the `lapply` loop are only part of the inefficiency. The deeper issue is that the algorithm repeatedly performs expensive string-based key generation and hash lookups for every row (â‰ˆ6.46M times), even though the neighbor relationships and year structure are highly regular. This is an **O(N Ã— avg_neighbors)** operation with heavy overhead from string manipulation and name-based indexing.

Since the neighbor structure is static across years, the current approach redundantly recomputes neighbor indices for each row-year combination. This is a **broader algorithmic inefficiency**: the neighbor lookup should be precomputed once for all years in integer form, avoiding string keys entirely.

---

### **Optimization Strategy**
1. **Precompute integer-based neighbor indices for all rows**:
   - Map `(id, year)` to row index using a fast integer matrix or vectorized approach.
   - Avoid string concatenation and named lookups inside the main loop.
2. **Exploit panel regularity**:
   - If every `id` appears in every year, then row index = `(id_position - 1) * n_years + year_position`.
   - Precompute a matrix of neighbor indices for each cell across all years.
3. **Vectorize neighbor stats computation**:
   - Use matrix operations instead of `lapply` where possible.

---

### **Working R Code**

```r
# Assumptions:
# - data$id and data$year are integers or can be coerced to integers
# - id_order is the vector of unique IDs in the desired order
# - years is the sorted vector of unique years
# - neighbors is a list of integer neighbor positions (spdep::nb)

optimize_neighbor_lookup <- function(data, id_order, neighbors) {
  # Precompute mappings
  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  id_to_pos   <- setNames(seq_along(id_order), id_order)
  year_to_pos <- setNames(seq_along(years), years)
  
  # Compute row index matrix: row_index[id_pos, year_pos]
  # Row index formula: (id_pos - 1) * n_years + year_pos
  row_index_matrix <- matrix(
    seq_len(nrow(data)),
    nrow = n_ids,
    ncol = n_years,
    byrow = TRUE
  )
  
  # Build neighbor lookup as a list of integer vectors
  # Each element corresponds to a row in data
  neighbor_lookup <- vector("list", nrow(data))
  
  for (id_pos in seq_len(n_ids)) {
    # Get neighbors for this id
    neigh_ids <- neighbors[[id_pos]]
    if (length(neigh_ids) == 0) next
    
    for (year_pos in seq_len(n_years)) {
      row_idx <- row_index_matrix[id_pos, year_pos]
      neigh_rows <- row_index_matrix[neigh_ids, year_pos]
      neighbor_lookup[[row_idx]] <- neigh_rows
    }
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Preallocate result matrix
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  
  colnames(result) <- c("max", "min", "mean")
  result
}

# Usage:
# Precompute neighbor lookup once
neighbor_lookup <- optimize_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Compute stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**
- Eliminates repeated string concatenation and hash lookups.
- Uses integer indexing and precomputed row index matrix.
- Neighbor lookup is built **once** for all years.
- Memory footprint is manageable: `neighbor_lookup` is a list of length â‰ˆ6.46M, but each element is a small integer vector (avg. 4â€“6 neighbors).

---

**Expected Performance Gain**:  
From 86+ hours to a few hours (or less) on a 16 GB laptop, as the main bottleneck (string-based repeated lookups) is removed and replaced with integer-based direct indexing.
END_RESPONSE>>>
