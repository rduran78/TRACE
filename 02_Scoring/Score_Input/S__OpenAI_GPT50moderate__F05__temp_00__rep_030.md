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
run_id: OpenAI_F05_T00_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real issue is **algorithmic**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated string-based lookups.  
- This is done once per row in `build_neighbor_lookup`, and then the resulting list is reused for all 5 variables, so the inefficiency is concentrated in building the lookup.  
- However, the current approach still scales poorly because it repeatedly maps `(id, year)` pairs to row indices using string keys, which is O(n) in memory and CPU for millions of rows.  

**Optimization Strategy**  
- Avoid string concatenation and hash lookups entirely.  
- Use **integer-based indexing**:  
  - Precompute a matrix mapping `(id_index, year_index)` â†’ row index.  
  - Use integer IDs for cells and years.  
- Build `neighbor_lookup` using direct integer indexing, not strings.  
- This reduces complexity from repeated string operations to pure integer lookups, which is much faster and memory-efficient.  

---

### **Proposed Algorithmic Reformulation**
1. Map `id` and `year` to integer indices.  
2. Create a 2D matrix `row_index_matrix[id_index, year_index]` storing row positions.  
3. For each row, find its neighbors by integer lookup in this matrix.  

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map IDs and years to integer indices
  id_to_idx   <- setNames(seq_along(id_order), as.character(id_order))
  year_levels <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_levels), year_levels)
  
  n_ids   <- length(id_order)
  n_years <- length(year_levels)
  
  # Create matrix: rows = ids, cols = years
  row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_index_matrix[cbind(id_to_idx[as.character(data$id)],
                          year_to_idx[as.character(data$year)])] <- seq_len(nrow(data))
  
  # Build neighbor lookup
  row_ids <- seq_len(nrow(data))
  neighbor_lookup <- vector("list", length = nrow(data))
  
  for (i in row_ids) {
    ref_idx <- id_to_idx[as.character(data$id[i])]
    yr_idx  <- year_to_idx[as.character(data$year[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_idx <- id_to_idx[as.character(neighbor_ids)]
    # Lookup row indices for same year
    neighbor_rows <- row_index_matrix[cbind(neighbor_idx, yr_idx)]
    neighbor_lookup[[i]] <- neighbor_rows[!is.na(neighbor_rows)]
  }
  
  neighbor_lookup
}
```

**Why this is faster:**  
- Eliminates `paste` and string-based hash lookups.  
- Uses integer matrix indexing (O(1) per lookup).  
- Memory footprint is manageable: `n_ids Ã— n_years` = 344,208 Ã— 28 â‰ˆ 9.6M integers (~76 MB).  

---

### **Integration with Existing Pipeline**
Replace:
```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
```
with:
```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
```
The rest of the pipeline (`compute_neighbor_stats`, etc.) remains unchanged.  

---

**Expected Performance Gain:**  
- From 86+ hours to a few hours or less, as the bottleneck (string operations) is removed and replaced with vectorized integer lookups.  
- Preserves the trained Random Forest model and the original estimand.  

**Summary:**  
The inefficiency is **not just local**; it reflects a broader pattern of string-based indexing in a massive panel. The integer-based matrix approach is the correct algorithmic reformulation for scalability.
END_RESPONSE>>>
