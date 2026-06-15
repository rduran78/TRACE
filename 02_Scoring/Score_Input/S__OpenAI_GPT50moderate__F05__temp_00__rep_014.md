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
run_id: OpenAI_F05_T00_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:  
- For each of ~6.46M rows, you repeatedly build `neighbor_keys` and perform string-based lookups.  
- This is done once per row in `build_neighbor_lookup`, and then the resulting neighbor index lists are reused for all 5 variables.  
- So the string work happens only once per row, but the cost is still huge because `paste` and hash lookups scale poorly at millions of iterations.  

**Broader issue:** The entire neighbor lookup could be precomputed as **integer indices** without string keys. The `id` and `year` combination can be mapped to a **dense integer index** using vectorized operations, eliminating repeated string concatenation and hash lookups.

---

### **Optimization Strategy**
1. **Precompute a fast integer mapping** from `(id, year)` â†’ row index using vectorized operations.
2. **Avoid string concatenation entirely** by using numeric encoding:  
   `global_index = (year_index - 1) * n_ids + id_index`
3. Build `neighbor_lookup` as a list of integer vectors directly.
4. Reuse this integer-based lookup for all variables without recomputation.

This reduces complexity from repeated string operations to pure integer indexing, which is orders of magnitude faster and memory-efficient.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Precompute mappings
  n_ids   <- length(id_order)
  n_rows  <- nrow(data)
  
  # Map id and year to integer positions
  id_to_pos   <- setNames(seq_along(id_order), as.character(id_order))
  year_levels <- sort(unique(data$year))
  year_to_pos <- setNames(seq_along(year_levels), year_levels)
  
  # Compute global index for each row
  id_pos   <- id_to_pos[as.character(data$id)]
  year_pos <- year_to_pos[as.character(data$year)]
  global_index <- (year_pos - 1L) * n_ids + id_pos
  
  # Build a direct lookup: global_index -> row index
  idx_lookup <- integer(max(global_index))
  idx_lookup[global_index] <- seq_len(n_rows)
  
  # Precompute neighbor lookup
  row_ids <- seq_len(n_rows)
  lapply(row_ids, function(i) {
    ref_idx <- id_to_pos[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_pos <- id_to_pos[as.character(neighbor_ids)]
    neighbor_global <- (year_pos[i] - 1L) * n_ids + neighbor_pos
    result <- idx_lookup[neighbor_global]
    as.integer(result[result > 0L])
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

### **Why This Works**
- Eliminates string concatenation and hash lookups.
- Uses integer arithmetic for O(1) mapping.
- Memory footprint is manageable: `idx_lookup` is a single integer vector of length `n_ids * n_years` (~9.6M entries, ~38 MB).
- The neighbor lookup is computed **once** and reused for all variables.

**Expected speedup:** From 86+ hours to a few hours or less, depending on I/O and downstream steps, because the bottleneck (string-based lookups) is removed.

**Preserves:**  
- Original numerical estimand.  
- Trained Random Forest model.  
- Panel structure and neighbor relationships.
END_RESPONSE>>>
