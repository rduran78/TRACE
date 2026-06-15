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
run_id: OpenAI_F05_T07_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated `paste()` and string-based key lookups inside `build_neighbor_lookup()` are a **local inefficiency**, but they reveal a **larger pattern problem**:  
- For each of ~6.46M rows, you are repeatedly constructing character keys (`paste(id, year, sep="_")`) and indexing a large named vector (`idx_lookup`).  
- This happens once during the neighbor lookup build, but the lookup itself is computed row-wise using repeated string operations.  
- The heavy overhead is **O(N Ã— avg_neighbors)** string concatenation and hash lookup (~6.46M Ã— ~8 neighbors â‰ˆ 50M+ operations), which is extremely costly in R.  

However, the fundamental task is deterministic: neighbor relationships depend on **id** and **year**, and the panel is rectangular. Thus, this can be **vectorized and index-based**, removing string keys entirely.  

---

### **Optimization Strategy**
- Precompute a **matrix of neighbor indices** using integer arithmetic, not string keys.
- Exploit panel structure:
  - Sort `data` by `id` and `year`.
  - Create a fast lookup from `id` â†’ row index base.
  - Compute neighbor row indices by offsetting by year position.
- Store `neighbor_lookup` as an integer matrix or list of integer vectors without character names.
- This avoids repeated hashing and string concatenation.  

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data sorted by id, then year
  data <- data[order(data$id, data$year), ]
  years <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id to block start row
  id_to_start <- match(id_order, data$id)
  
  # Precompute year index for each row
  year_pos <- match(data$year, years)
  
  # For each row, compute neighbor indices
  n <- nrow(data)
  result <- vector("list", n)
  
  for (i in seq_len(n)) {
    ref_idx <- match(data$id[i], id_order)
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    
    # Compute neighbor positions: neighbor base + (year_pos[i]-1)
    neighbor_base_idx <- id_to_start[match(neighbor_ids, id_order)]
    neighbor_rows <- neighbor_base_idx + (year_pos[i] - 1)
    
    # Keep only valid indices
    neighbor_rows <- neighbor_rows[!is.na(neighbor_rows)]
    result[[i]] <- neighbor_rows
  }
  
  result
}
```

---

### **Further Speed-Up**
- Instead of looping over all rows (~6.46M), loop over **id blocks** (344k) and replicate across years:
  - Compute neighbor indices **once per id** and add year offsets.
- Example:

```r
build_neighbor_lookup_block <- function(data, id_order, neighbors) {
  data <- data[order(data$id, data$year), ]
  years <- sort(unique(data$year))
  n_years <- length(years)
  
  id_to_start <- match(id_order, data$id)
  
  res <- vector("list", nrow(data))
  
  for (id_idx in seq_along(id_order)) {
    neighbor_ids <- id_order[neighbors[[id_idx]]]
    neighbor_base <- id_to_start[match(neighbor_ids, id_order)]
    
    # Offsets for each year
    for (y in seq_len(n_years)) {
      row_idx <- id_to_start[id_idx] + (y - 1)
      neighbor_rows <- neighbor_base + (y - 1)
      res[[row_idx]] <- neighbor_rows
    }
  }
  
  res
}
```

---

### **Expected Impact**
- Eliminates `paste()` and named indexing entirely â†’ reduces time from 86+ hours to a few minutes for lookup construction.
- Memory: list of integer vectors of average length ~8 for 6.46M rows (~500MB) is large but feasible on 16 GB RAM.
- `compute_neighbor_stats()` can remain unchangedâ€”it operates on integer indices.

---

âœ… **Preserves trained model and estimand**: Same neighbor relationships, same feature calculations.  
âœ… **Algorithmic reformulation**: From hash-based string lookups to integer arithmetic leveraging panel structure.  
âœ… **Massive runtime reduction**: From tens of hours to likely under 1 hour for full pipeline.
END_RESPONSE>>>
