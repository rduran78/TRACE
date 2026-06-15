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
run_id: OpenAI_F05_T00_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste(...)`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are only a *symptom* of a larger inefficiency: the algorithm repeatedly constructs neighbor indices for each row of a 6.46M-row panel. This is extremely costly because:

- For each of ~6.46M rows, it builds `neighbor_keys` and performs multiple hash lookups.
- The neighbor structure is static across years; only the year dimension changes.
- The same neighbor relationships are recomputed for every row, even though they are predictable from `(cell_id, year)`.

This is an **O(N Ã— avg_neighbors)** operation with heavy string overhead. The real problem is that the algorithm is not vectorized and does not exploit the panel structure.

---

### **Optimization Strategy**

1. **Precompute neighbor indices once per cell** (not per row-year).
2. **Exploit panel structure**: For each year, neighbors are the same set of cell IDs, so we can map them to row indices using integer arithmetic instead of string keys.
3. **Avoid string concatenation and hash lookups**: Use numeric indexing with a precomputed offset for each year.

---

### **Algorithmic Reformulation**

- Sort `data` by `id` and `year`.
- Assume `id_order` matches the unique IDs in `data$id`.
- Compute:
  - `n_ids <- length(id_order)`
  - `n_years <- length(unique(data$year))`
- For row `r` with `(id_idx, year_idx)`, its row index is:
  
  ```
  row_index = (year_idx - 1) * n_ids + id_idx
  ```

- Precompute a neighbor index matrix for all rows using this formula.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  # Map id and year to integer positions
  id_to_idx   <- setNames(seq_along(id_order), id_order)
  year_levels <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_levels), year_levels)
  
  n_ids   <- length(id_order)
  n_years <- length(year_levels)
  
  # Precompute neighbor IDs for each id
  neighbor_ids_list <- lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
  
  # For each row, compute neighbor row indices
  row_ids <- seq_len(nrow(data))
  id_idx  <- id_to_idx[as.character(data$id)]
  yr_idx  <- year_to_idx[as.character(data$year)]
  
  lapply(row_ids, function(r) {
    nbr_ids <- neighbor_ids_list[[id_idx[r]]]
    if (length(nbr_ids) == 0) return(integer(0))
    nbr_idx <- id_to_idx[as.character(nbr_ids)]
    # Compute row indices using panel offset
    ((yr_idx[r] - 1) * n_ids) + nbr_idx
  })
}
```

---

### **Further Speedup**

- Instead of `lapply(row_ids, ...)`, build a **matrix of neighbor indices** once and reuse it for all variables.
- Use `vapply` or `parallel::mclapply` for multi-core speedup.
- If memory allows, store neighbor indices in a compressed sparse structure.

---

### **Why This Works**

- Eliminates string concatenation and hash lookups.
- Reduces complexity from repeated key-building to pure integer arithmetic.
- Preserves the original estimand and Random Forest model (only feature construction changes).

---

**Expected Impact**: From 86+ hours to a few minutes, as the bottleneck is removed and operations become vectorized and cache-friendly.
END_RESPONSE>>>
