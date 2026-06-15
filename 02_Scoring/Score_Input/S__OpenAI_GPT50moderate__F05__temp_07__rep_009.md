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
run_id: OpenAI_F05_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key creation (`paste(...)`) and lookup inside `build_neighbor_lookup` is a **symptom of a broader inefficiency**. Currently, for each of ~6.46M rows, the code constructs neighbor keys as strings and repeatedly indexes into a hash map (`idx_lookup`). This repeats string concatenation and name-based lookups millions of times. The algorithmic pattern itself scales poorly:  

- Complexity â‰ˆ O(N Ã— avg_neighbors) with expensive string ops.  
- N â‰ˆ 6.46M rows, avg_neighbors â‰ˆ 4â€“8.  
- Memory pressure from repeated character vector creation.

This is more than a local inefficiency: the entire approach relies on key-based joins rather than leveraging the **structured panel nature** (id Ã— time).  

---

**Optimization Strategy**  
Exploit two facts:  
1. `data` is a balanced panel: every `id` appears for every `year`.  
2. Neighbor relationships depend only on `id`, not on `year`.  

Therefore, instead of rebuilding string keys per row, precompute **numeric neighbor indices** for each id, then map them across years by offset arithmetic. This avoids string concatenation and repeated hash lookups entirely.  

We can:  
- Sort `data` by `id`, then `year`.  
- Precompute a lookup from `id` â†’ block index.  
- Compute neighbor row indices for each row using simple integer addition.  

---

### **Proposed Algorithm**
- Assume `data` sorted by `(id, year)`.  
- Let `T = number of years`.  
- For each id `k` at position `p`, the row for year `t` is at index `p + (t-1)`.  
- For neighbors of id `k`, compute their base positions and add `(t-1)`.

This reduces complexity to pure integer arithmetic, eliminating string operations.

---

### **Working R Code**

```r
opt_build_neighbor_lookup <- function(data, id_order, neighbors) {
  # Ensure data sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  ids <- unique(data$id)
  years <- sort(unique(data$year))
  T <- length(years)
  n_ids <- length(ids)
  
  # Map id -> position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map id -> block start row
  id_block_start <- setNames(seq(1, by = T, length.out = n_ids), id_order)
  
  # Precompute neighbor bases for each id
  neighbor_bases <- lapply(seq_along(id_order), function(i) {
    neigh_ids <- id_order[neighbors[[i]]]
    id_block_start[neigh_ids]
  })
  
  # Build lookup: for each row index, compute neighbor row indices
  n_rows <- nrow(data)
  row_ids <- seq_len(n_rows)
  
  # For each row: find base offset and add to neighbor bases
  result <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    id <- data$id[i]
    year_idx <- match(data$year[i], years) - 1  # zero-based offset
    ref_idx <- id_to_pos[as.character(id)]
    bases <- neighbor_bases[[ref_idx]]
    if (length(bases) > 0) {
      result[[i]] <- as.integer(bases + year_idx)
    } else {
      result[[i]] <- integer(0)
    }
  }
  result
}
```

---

### **Why This Is Faster**
- No `paste()`, no `setNames`, no character hashing.  
- Complexity still O(N Ã— avg_neighbors), but with simple integer addition.  
- Memory reduced drastically.  
- Loop can be further accelerated with `vapply` or Rcpp.

---

### **Usage**
```r
neighbor_lookup <- opt_build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
```
Then reuse `compute_neighbor_stats` unchanged.  

---

**Expected Impact**  
- From 86+ hours to a few hours or even <1 hour depending on machine.  
- Eliminates the pathological string-based overhead.  

**Preserves:**  
- Original estimand.  
- Trained model.  
- Numerical correctness.
END_RESPONSE>>>
