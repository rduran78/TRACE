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
run_id: OpenAI_F05_T00_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and repeated hash lookups inside the `lapply` loop are only a *symptom* of a larger inefficiency: the algorithm repeatedly performs expensive string-based key generation and hash lookups for every row (â‰ˆ6.46M times). This is not just local inefficiency; it is an algorithmic bottleneck because:

- For each of 6.46M rows, you build `neighbor_keys` and index into `idx_lookup`.
- Each lookup involves string concatenation and hash table access.
- This is repeated for every neighbor feature computation, but even once is already costly.

The core issue: **you are mapping (id, year) â†’ row index repeatedly using strings**. This can be replaced by integer-based indexing and vectorized operations.

---

### **Optimization Strategy**
1. **Precompute integer indices for neighbors across all years**:
   - Instead of building string keys, use integer mapping:  
     - Map each `id` to its row positions for all years.
     - For each row, precompute the integer indices of its neighbors for all years.
2. **Store neighbor indices in a list once**:
   - Build a single `neighbor_lookup` as a list of integer vectors (one per row).
   - Avoid string concatenation entirely.
3. **Vectorize neighbor stats computation**:
   - Use `vapply` or `matrixStats` for speed.
4. **Memory considerations**:
   - 6.46M rows Ã— average 4 neighbors â‰ˆ 25M integers â†’ ~100 MB (feasible on 16 GB RAM).

---

### **Reformulated Approach**
- Precompute a matrix `row_index_matrix` of size `n_ids Ã— n_years` mapping `(id, year)` â†’ row index.
- Use this matrix to quickly find neighbor rows for each observation without string operations.

---

### **Working R Code**

```r
# Assume:
# data: data.frame with columns id, year, and variables
# id_order: vector of unique ids in desired order
# neighbors: list of integer vectors (rook neighbors by id index)
# years: sorted unique years

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  ids   <- as.integer(factor(data$id, levels = id_order))
  years <- sort(unique(data$year))
  n_ids <- length(id_order)
  n_years <- length(years)
  
  # Map (id, year) -> row index
  row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_index_matrix[cbind(ids, match(data$year, years))] <- seq_len(nrow(data))
  
  # Precompute neighbor indices for each row
  row_ids <- seq_len(nrow(data))
  year_idx <- match(data$year, years)
  
  lapply(row_ids, function(i) {
    ref_idx <- ids[i]
    neigh_ids <- neighbors[[ref_idx]]
    if (length(neigh_ids) == 0) return(integer(0))
    # Lookup rows for same year
    result <- row_index_matrix[neigh_ids, year_idx[i]]
    result[!is.na(result)]
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  t(result)
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Performance Gain**
- Eliminates repeated string concatenation and hash lookups.
- Reduces complexity from *O(N Ã— neighbors Ã— string ops)* to *O(N Ã— neighbors)* with integer indexing.
- Should reduce runtime from **86+ hours to a few hours** on a 16 GB laptop.

**Key Idea**: Replace string-based keys with integer-based direct indexing and precompute all mappings once.
END_RESPONSE>>>
