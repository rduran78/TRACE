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
run_id: OpenAI_F05_T03_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated `idx_lookup` and `neighbor_keys` construction inside the `lapply` loop is only a **local inefficiency symptom of a larger repeated lookup pattern**. Specifically:

- For each of ~6.46M rows, you:
  - Recompute `neighbor_keys` by pasting strings for all neighbors.
  - Perform repeated hash lookups in `idx_lookup`.
- This is repeated for every variable in `neighbor_source_vars`, but even before that, `build_neighbor_lookup` itself does this string concatenation for every row.
- The real issue: **string-based keys for every lookup** across millions of rows is extremely costly in both time and memory.  
- The panel structure is regular: each cell has 28 years. The neighbor relationships are static across years. So the neighbor index mapping can be computed **once** in integer space and reused.

**Optimization Strategy**  
- Avoid string concatenation and hash lookups entirely.
- Precompute:
  - A mapping from `id` to row index for each year using integer arithmetic.
  - A global neighbor index structure that works across all years.
- Use matrix indexing or integer offsets instead of string keys.
- Build a single integer matrix `neighbor_lookup` of length = nrow(data), each element a list of integer indices for neighbors.  
- Then reuse this for all variables without recomputing anything.

**Algorithmic Reformulation**  
- Sort `data` by `id` and `year` so that rows for each `id` are contiguous.
- Compute `n_years <- length(unique(data$year))`.
- For each row `i`:
  - Find its `id_idx` (position in `id_order`).
  - Get its neighborsâ€™ `id_idx` from `neighbors[[id_idx]]`.
  - Compute neighbor row indices as `(neighbor_id_idx - 1) * n_years + year_idx`.
- This avoids string operations and repeated hashing.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id and year
  data <- data[order(data$id, data$year), ]
  row_ids <- seq_len(nrow(data))
  
  # Precompute mappings
  n_years <- length(unique(data$year))
  year_levels <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_levels), year_levels)
  
  id_to_idx <- setNames(seq_along(id_order), id_order)
  id_idx_vec <- id_to_idx[as.character(data$id)]
  year_idx_vec <- year_to_idx[as.character(data$year)]
  
  # Build neighbor lookup using integer offsets
  lapply(row_ids, function(i) {
    ref_idx <- id_idx_vec[i]
    neighbor_ids <- neighbors[[ref_idx]]
    if (length(neighbor_ids) == 0) return(integer(0))
    neighbor_id_idx <- neighbor_ids
    # Compute row indices for same year
    neighbor_rows <- (neighbor_id_idx - 1) * n_years + year_idx_vec[i]
    neighbor_rows
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  colnames(result) <- c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean"))
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why this works:**
- Eliminates string concatenation and hash lookups.
- Uses integer arithmetic for O(1) neighbor index computation.
- Precomputes everything once and reuses it.
- Memory footprint: `neighbor_lookup` is a list of integer vectors, same as before but built efficiently.

**Expected Performance Gain:**  
From 86+ hours to a few hours or less, because:
- String-based operations on 6.46M rows Ã— multiple neighbors Ã— multiple variables are replaced by integer math.
- No repeated lookups for each variable.

This preserves:
- Original numerical estimand.
- Trained Random Forest model.
- Panel structure and neighbor logic.
END_RESPONSE>>>
