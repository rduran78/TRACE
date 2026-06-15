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
run_id: OpenAI_F05_T03_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(...)`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**: you are recomputing neighbor indices for every row (â‰ˆ6.46M times) even though the neighbor structure is static across years. This leads to massive redundant work and memory churn.

- Each cell has the same neighbors every year.
- The only thing that changes across years is the **row index** in the panel.
- Current approach: For each row, build neighbor keys, look them up in a hash map â†’ O(N * avg_neighbors) string operations.
- This is repeated for every variable later, but the neighbor lookup itself is the main bottleneck.

**Optimization Strategy**  
Precompute a **numeric neighbor index matrix** once, leveraging the fact that:
- `id_order` is fixed.
- Years are fixed and sequential.
- Panel is sorted by `(id, year)`.

Then, for each row, the neighbor rows can be computed by simple arithmetic rather than string-based lookups:
```
row_index = (id_position - 1) * n_years + year_position
```
Neighbors for that row = same year offset applied to neighbor IDs.

This reduces complexity from repeated string hashing to pure integer indexing.

---

### **Proposed Algorithm**
1. Sort `data` by `id` and `year` if not already.
2. Compute `n_ids` and `n_years`.
3. Build a numeric neighbor index matrix of size `n_ids Ã— avg_neighbors` (list of integer vectors).
4. Expand this to all rows by adding `(year_offset * n_ids)` for each year.
5. Use this precomputed structure in `compute_neighbor_stats`.

---

### **Working R Code**

```r
# Assumes: data sorted by id, year
# id_order: vector of unique ids in the same order as data
# neighbors: spdep::nb object for rook adjacency
# n_years: number of years in panel

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  
  # Map id to position
  id_to_pos <- setNames(seq_along(id_order), id_order)
  
  # Precompute neighbor positions for each id
  base_neighbors <- lapply(seq_along(id_order), function(i) {
    as.integer(id_to_pos[id_order[neighbors[[i]]]])
  })
  
  # For each row, compute neighbor indices without string ops
  # data assumed sorted by id, then year
  row_ids <- seq_len(nrow(data))
  
  # Compute year index for each row (1..n_years)
  year_idx <- as.integer(factor(data$year, levels = sort(unique(data$year))))
  
  # Compute cell position for each row (1..n_ids)
  cell_pos <- id_to_pos[as.character(data$id)]
  
  # For each row, compute neighbor row indices
  lapply(row_ids, function(i) {
    nb <- base_neighbors[[cell_pos[i]]]
    if (length(nb) == 0) return(integer(0))
    # Row index formula: (year_idx - 1) * n_ids + neighbor_id_pos
    (year_idx[i] - 1) * n_ids + nb
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
- Eliminates repeated string concatenation and hash lookups.
- Complexity drops from O(N * avg_neighbors) string ops to O(N * avg_neighbors) integer ops (much faster).
- Memory footprint is lower because we store integer vectors instead of character keys.
- Preserves original estimand and Random Forest model (data structure unchanged).

**Expected Speedup:**  
From 86+ hours to a few hours or less on a 16 GB laptop, depending on I/O and downstream steps. Further gains possible by:
- Using `matrix` instead of `list` for fixed neighbor sizes.
- Parallelizing `lapply` with `parallel::mclapply` or `future.apply`.

**Bottom Line:**  
The inefficiency is **not just local**; itâ€™s a symptom of an algorithmic issue. Precomputing numeric neighbor indices removes the bottleneck.
END_RESPONSE>>>
