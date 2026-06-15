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
run_id: OpenAI_F05_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The repeated `paste()` and string-based lookups inside `build_neighbor_lookup()` are **only a local inefficiency, but symptomatic of a broader algorithmic issue**. The current design repeatedly maps `(id, year)` â†’ row index via a character-keyed `idx_lookup`. This involves:

- Constructing millions of string keys (`id_year`) and hashing them.
- Doing this for **every row** across ~6.46M observations.
- Repeating the computation for each of the 5 neighbor source variables.

This is extremely expensive because the lookup is recalculated for each row and depends on string processing rather than integer indexing. The underlying problem is that the algorithm is operating at the row level instead of exploiting the **panel structure** (id Ã— year) and **static neighbor relationships**.

---

### **Optimization Strategy**
- **Precompute an integer-based neighbor index matrix once** for all rows, eliminating repeated string concatenation and hash lookups.
- Use **matrix operations** or `vapply` instead of repeated `lapply` calls with dynamic string handling.
- Leverage the fact that `neighbors` is static across years: for each cell `id`, its neighbors in each year correspond to a fixed set of row offsets.
- Build a **neighbor index matrix** of size `nrow(data)` Ã— `max_degree` (fill with `NA` for missing neighbors). Then reuse this for all variables.

This reduces complexity from repeated string-based hashing to pure integer lookups, cutting runtime from days to hours (or minutes).

---

### **Reformulated R Code**

```r
# Precompute neighbor index matrix once
build_neighbor_index_matrix <- function(data, id_order, neighbors) {
  # Map id -> row indices by year
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  id_to_rows <- split(seq_len(nrow(data)), data$id)  # list: id -> row indices

  max_deg <- max(lengths(neighbors))
  n_rows  <- nrow(data)
  
  # Preallocate integer matrix (n_rows x max_deg)
  neighbor_matrix <- matrix(NA_integer_, nrow = n_rows, ncol = max_deg)
  
  for (ref_idx in seq_along(id_order)) {
    ref_id <- id_order[ref_idx]
    ref_rows <- id_to_rows[[as.character(ref_id)]]
    if (is.null(ref_rows)) next
    
    # Neighbor IDs for this cell
    nb_ids <- id_order[neighbors[[ref_idx]]]
    if (length(nb_ids) == 0) next
    
    # For each year (row in ref_rows), map to neighbor rows of same year
    for (rpos in seq_along(ref_rows)) {
      ref_row <- ref_rows[rpos]
      yr <- data$year[ref_row]
      
      # Find neighbor rows for this year
      nb_rows <- integer(0)
      for (nb_id in nb_ids) {
        nb_set <- id_to_rows[[as.character(nb_id)]]
        if (!is.null(nb_set)) {
          # Find row for this year (same index as yr)
          # Assuming each id has one row per year in same order
          nb_row <- nb_set[rpos]  # position matches year ordering
          nb_rows <- c(nb_rows, nb_row)
        }
      }
      if (length(nb_rows) > 0) {
        neighbor_matrix[ref_row, seq_along(nb_rows)] <- nb_rows
      }
    }
  }
  
  neighbor_matrix
}

# Compute neighbor stats using precomputed matrix
compute_neighbor_stats_matrix <- function(data, neighbor_matrix, var_name) {
  vals <- data[[var_name]]
  apply(neighbor_matrix, 1, function(idx) {
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }) |> t()
}

# Pipeline
neighbor_matrix <- build_neighbor_index_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_matrix(cell_data, neighbor_matrix, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why this works**
- **No repeated string concatenation or hashing.**
- Precomputes neighbor indices once using integer positions.
- Each neighbor set is reused across all variables.
- Reduces complexity from O(n Ã— neighbors Ã— vars) with hashing to O(n Ã— neighbors + n Ã— vars).

This approach should **cut runtime from 86+ hours to a few hours or less** on a 16â€¯GB laptop, without changing the estimand or retraining the Random Forest model.
END_RESPONSE>>>
