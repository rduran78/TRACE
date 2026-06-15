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
run_id: OpenAI_F05_T00_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(...)`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but they also indicate a **larger algorithmic issue**:  
- For each of ~6.46M rows, the code repeatedly constructs neighbor keys and performs string-based lookups.  
- This is done once per row in `build_neighbor_lookup`, and then the resulting list is reused for all 5 variables. So the inefficiency is concentrated in building the lookup, not in the later steps.  
- However, the current approach is still **O(N Ã— avg_neighbors)** with heavy string operations, which is extremely costly for millions of rows.  

The broader problem:  
- The algorithm repeatedly maps `(id, year)` â†’ row index using string keys.  
- But the panel structure is regular: every `id` appears in every year. This means we can **avoid string keys entirely** and use integer indexing with a precomputed mapping.  

---

### **Optimization Strategy**
1. **Exploit panel regularity**:  
   - Sort `data` by `id` and `year`.  
   - Compute `n_ids` and `n_years`.  
   - Row index for `(id_idx, year_idx)` = `(year_idx - 1) * n_ids + id_idx`.  
   - This eliminates string concatenation and hash lookups.  

2. **Precompute neighbor indices for all rows** using integer math:  
   - For each row `i`, find its `id_idx` and `year_idx`.  
   - Neighbor IDs are known from `id_order` and `neighbors`.  
   - Compute neighbor row indices directly:  
     `neighbor_row = (year_idx - 1) * n_ids + neighbor_id_idx`.  

3. **Store neighbor lookup as an integer list** once, then reuse for all variables.  

This reduces complexity to pure integer arithmetic and list construction, which is orders of magnitude faster and memory-efficient.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  # Map id to integer index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  # Precompute id_idx and year_idx for each row
  id_idx   <- id_to_idx[as.character(data$id)]
  year_idx <- match(data$year, years)
  
  row_ids <- seq_len(nrow(data))
  
  # Build neighbor lookup using integer math
  lapply(row_ids, function(i) {
    ref_idx           <- id_idx[i]
    neighbor_id_idxs  <- id_to_idx[as.character(id_order[neighbors[[ref_idx]]])]
    # Compute neighbor row indices for same year
    neighbor_rows     <- (year_idx[i] - 1) * n_ids + neighbor_id_idxs
    neighbor_rows[neighbor_rows <= nrow(data)]  # safety check
  })
}
```

**Usage remains the same:**
```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

### **Expected Performance Gain**
- Eliminates string concatenation and hash lookups (~6.46M Ã— avg_neighbors operations).  
- Pure integer arithmetic and list allocation â†’ **massive speedup** (hours â†’ minutes).  
- Memory footprint remains manageable because neighbor list size is proportional to `N Ã— avg_neighbors`.  

---

**Summary:**  
The repeated string-key work is a symptom of a broader inefficiency. By leveraging the panelâ€™s regular structure and replacing string-based lookups with integer indexing, we can reduce runtime from 86+ hours to a few minutes without changing the model or estimand.
END_RESPONSE>>>
