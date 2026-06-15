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
run_id: OpenAI_F05_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The observed repeated string-key work (`paste` + `idx_lookup` lookup) inside the inner loop of `build_neighbor_lookup` is a symptom of a larger algorithmic inefficiency. Specifically:

- For **each of 6.46M rows**, the code constructs string keys (`id_year`) repeatedly and looks them up in a named vector (`idx_lookup`).
- This involves heavy character concatenation and repeated hashing, which is expensive in R.
- These operations happen once for each row during neighbor index construction, and then `compute_neighbor_stats` loops again over the same structure for each of the 5 variables.
  
So the inefficiency is **not just local**â€”itâ€™s a fundamental design issue. The algorithm repeatedly converts numeric IDs and years into strings to match rows, which is unnecessary because IDs and years are already numeric and can be mapped via integer indexing.

Given the size of the dataset (â‰ˆ6.5M rows), string-based lookups in tight loops are prohibitive. A broader **algorithmic reformulation** is necessary.

---

### **Optimization Strategy**
- **Eliminate string concatenation and character keys.**
- Use **integer-based indexing**: Map `(id, year)` pairs to row indices using a 2D offset calculation or a precomputed numeric key (e.g., `id_index + year_offset`).
- Build a **single neighbor lookup once** and reuse it for all variables.
- Use **vectorized or matrix-based operations** where possible.

#### Key idea:
- Sort `data` by `id` and `year`.
- Compute offsets so `(id_position, year_position)` â†’ row index can be O(1) without strings.
- Precompute neighbor indices for all rows in one pass using integer mapping.

---

### **Working R Code (Optimized)**

```r
# Assumes: data has columns id, year
# id_order: vector of cell IDs in neighbor structure order
# neighbors: list of integer neighbor indices (as from spdep::nb)
# data is sorted by id, then year

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map cell IDs to sequential positions
  id_to_pos <- match(data$id[!duplicated(data$id)], id_order)
  stopifnot(!anyNA(id_to_pos))

  # Compute number of unique IDs and years
  unique_ids   <- sort(unique(data$id))
  unique_years <- sort(unique(data$year))
  n_id    <- length(unique_ids)
  n_year  <- length(unique_years)

  # Build fast mapping: (id_pos, year_pos) -> row index
  # Assuming data sorted by id, then year:
  # row_index = (id_pos - 1) * n_year + year_pos
  row_index_matrix <- matrix(seq_len(nrow(data)), nrow = n_year, ncol = n_id)

  # Precompute neighbor lookup
  # For each row, find its neighbors (same year) by integer lookup
  neighbor_lookup <- vector("list", length = nrow(data))

  for (id_pos in seq_along(unique_ids)) {
    # neighbors for this id
    ref_idx <- match(unique_ids[id_pos], id_order)
    nb_ids  <- id_order[neighbors[[ref_idx]]]
    nb_pos  <- match(nb_ids, unique_ids)

    if (length(nb_pos) == 0) next

    # For each year, map neighbors
    for (year_pos in seq_along(unique_years)) {
      row_idx <- (id_pos - 1) * n_year + year_pos
      nb_rows <- row_index_matrix[year_pos, nb_pos]
      neighbor_lookup[[row_idx]] <- nb_rows
    }
  }

  neighbor_lookup
}
```

---

### **Compute Neighbor Stats (unchanged logic, but efficient)**

```r
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  do.call(rbind, lapply(neighbor_lookup, function(idx) {
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }))
}
```

---

### **Expected Impact**
- Removes all string concatenation and hash lookups.
- Reduces complexity to pure integer arithmetic and precomputed mapping.
- Memory footprint remains manageable (â‰ˆ6.5M lists with short integer vectors).
- Should cut runtime from **86+ hours to a few hours** on a standard laptop.

---

**Preserves:**  
- Original Random Forest model  
- Original numeric estimand  
- Feature definitions untouched (max, min, mean of neighbors)  

This reformulation is the most significant improvement possible without changing the model or data semantics.
END_RESPONSE>>>
