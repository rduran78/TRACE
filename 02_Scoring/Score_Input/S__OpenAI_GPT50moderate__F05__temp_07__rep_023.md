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
run_id: OpenAI_F05_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated construction of string keys using `paste()` and repeated lookups in `idx_lookup` inside `lapply(row_ids, ...)` represents a **local inefficiency**, but the issue is symptomatic of a **larger algorithmic pattern**:  
- For **every row** (~6.46 million), the code builds character keys for all neighbors and performs name-based indexing on `idx_lookup`.  
- This is repeated for every variable in `neighbor_source_vars`, although `neighbor_lookup` is precomputed, so the main waste is during `build_neighbor_lookup`.  
- The work is inherently O(N Ã— average_neighbors), dominated by string manipulation and hash lookups.  

Given that the neighbor structure is static across years and panel sizes are huge, the optimal approach is to **precompute neighbor row indices without string keys** and reuse them for all variables.  

---

### **Optimization Strategy**
1. **Avoid string keys entirely**: Instead of concatenating `id` and `year`, map `(id, year)` â†’ row index once using integer-based structures.
2. **Precompute `neighbor_lookup` efficiently**:
   - Convert `id` and `year` to integer factors.
   - Use vectorized matching or direct array indexing rather than per-row string operations.
3. **Preserve trained Random Forest model and estimand**: Only the feature-construction step changes; predictions remain unaffected.
4. **Memory-conscious approach**: Use integer matrices/lists instead of character vectors; avoid large `setNames()` calls.

---

### **Reformulated Algorithm**
- Create a 2D index mapping: `row_idx_matrix[id_idx, year_idx] = row number`.
- For each row, retrieve neighbors by integer lookup without string concatenation.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map IDs and years to integer positions
  id_levels   <- as.integer(factor(data$id, levels = id_order))
  year_levels <- as.integer(factor(data$year, levels = sort(unique(data$year))))
  n_id        <- length(id_order)
  n_year      <- length(unique(data$year))
  
  # Build a matrix: [id_idx, year_idx] -> row index
  row_idx_matrix <- matrix(NA_integer_, nrow = n_id, ncol = n_year)
  row_idx_matrix[cbind(id_levels, year_levels)] <- seq_len(nrow(data))
  
  # Precompute neighbor row indices
  row_ids <- seq_len(nrow(data))
  lapply(row_ids, function(i) {
    ref_idx           <- id_levels[i]
    y_idx             <- year_levels[i]
    neighbor_ids      <- neighbors[[ref_idx]]
    if (length(neighbor_ids) == 0) return(integer(0))
    neighbor_rows     <- row_idx_matrix[neighbor_ids, y_idx]
    as.integer(neighbor_rows[!is.na(neighbor_rows)])
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

# Pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

### **Why This Works**
- **Eliminates string concatenation and name-based lookups** (major performance bottleneck).
- **Uses direct integer indexing** via `row_idx_matrix`, making lookups O(1) and vectorized.
- Precompute once; reuse across all variables â†’ no repeated expensive operations.

---

**Expected Performance Gain**  
- Original approach: ~86 hours (due to repeated paste and hash lookups on millions of rows).
- New approach: **should reduce runtime by 1â€“2 orders of magnitude**, making it feasible on a 16 GB laptop.
END_RESPONSE>>>
