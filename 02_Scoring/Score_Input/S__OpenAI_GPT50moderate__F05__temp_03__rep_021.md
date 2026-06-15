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
run_id: OpenAI_F05_T03_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste`) and name-based indexing inside `build_neighbor_lookup` is not just a local inefficiencyâ€”itâ€™s part of a broader algorithmic pattern that repeatedly converts IDs and years into strings and performs hash lookups for every row and neighbor. With 6.46M rows and 1.37M neighbor relationships, this string-based approach is extremely costly in both time and memory.

The inefficiency stems from:
- Constructing `neighbor_keys` for every row using `paste()`.
- Using a named vector (`idx_lookup`) for lookups by string keys.
- Repeating this for each row in a large panel.

**Optimization Strategy**  
Reformulate the lookup to use **integer-based indexing** instead of string keys. Precompute:
- A mapping from `id` to an integer index (already in `id_order`).
- A mapping from `(id_index, year)` to the row index in `data`.

Then, for each row, compute neighbor indices using integer arithmetic, avoiding string operations entirely.

**Approach**:
1. Encode `id` and `year` as integers.
2. Compute a fast lookup table: `row_index_matrix[id_index, year_index] = row number`.
3. Use this matrix to retrieve neighbor rows in O(1) time without string concatenation.

---

### **Optimized R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Encode IDs and years as integers
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  year_vals <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_vals), year_vals)
  
  n_id <- length(id_order)
  n_year <- length(year_vals)
  
  # Create a matrix for fast lookup: rows = id, cols = year
  row_lookup <- matrix(NA_integer_, nrow = n_id, ncol = n_year)
  
  # Fill lookup matrix
  id_idx <- id_to_idx[as.character(data$id)]
  year_idx <- year_to_idx[as.character(data$year)]
  row_lookup[cbind(id_idx, year_idx)] <- seq_len(nrow(data))
  
  # Precompute neighbor lookup
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_idx[as.character(data$id[i])]
    y_idx   <- year_idx[i]
    neighbor_ids <- neighbors[[ref_idx]]
    neighbor_rows <- row_lookup[neighbor_ids, y_idx]
    as.integer(neighbor_rows[!is.na(neighbor_rows)])
  })
}
```

---

### **Compute Neighbor Stats (unchanged)**

```r
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
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
```

---

### **Expected Performance Gains**
- Eliminates string concatenation and hash lookups.
- Reduces complexity from repeated string operations to pure integer indexing.
- Memory footprint is manageable: `row_lookup` is `n_id Ã— n_year` (â‰ˆ 344k Ã— 28 â‰ˆ 9.6M integers â‰ˆ 76 MB).
- Should cut runtime from **86+ hours to a few hours** on a 16 GB machine.

---

**Key Properties Preserved**:
- Same neighbor relationships.
- Same numerical estimands.
- No retraining of the Random Forest model.

This is a **major algorithmic improvement**, not just a micro-optimization.
END_RESPONSE>>>
