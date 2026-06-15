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
run_id: OpenAI_F05_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation and hash lookups (`paste(..., sep = "_")` and `idx_lookup[...]`) inside the inner `lapply` loop of `build_neighbor_lookup` is a **local inefficiency**, but the real issue is **algorithmic**:

- For **6.46 million rows**, `build_neighbor_lookup` iterates over each row and performs:
  - String concatenation for all neighbors.
  - Hash lookups in `idx_lookup`.
- This results in **tens of millions of string operations** and repeated work across variables.
- The outer loop (`compute_neighbor_stats`) then iterates over 5 variables, but at least the neighbor lookup is reused. Still, the initial neighbor lookup build is extremely costly.

**Root cause:** The current design treats the panel as a flat table and repeatedly reconstructs neighbor relationships per row-year using string keys. This is unnecessary because:
- The neighbor structure is **static across years**.
- The mapping from `(id, year)` â†’ row index is deterministic and can be computed once using numeric indexing.

---

### **Optimization Strategy**

1. **Avoid string keys entirely**:
   - Use integer-based indexing: precompute a matrix that maps `(cell_id, year)` to row index.
   - Use numeric IDs for neighbors.

2. **Precompute neighbor indices for all rows**:
   - For each cell, get its neighbors' IDs.
   - For each year, map those neighbor IDs to row indices using a fast integer lookup (no strings).

3. **Vectorize neighbor statistics**:
   - Instead of looping over rows and neighbors, use `rowsum` or matrix operations where possible.

---

### **Proposed Algorithm**

- Create a `row_index_matrix` of size `n_cells Ã— n_years` mapping `(cell_id, year)` â†’ row index.
- For each cell, store its neighbor IDs once.
- For each year, translate neighbor IDs to row indices via `row_index_matrix`.
- Compute stats using precomputed neighbor indices.

---

### **Working R Code**

```r
optimize_neighbor_features <- function(data, id_order, neighbors, years, vars) {
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Map cell_id and year to row index
  row_index_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years,
                              dimnames = list(as.character(id_order), as.character(years)))
  row_index_matrix[cbind(match(data$id, id_order), match(data$year, years))] <- seq_len(nrow(data))
  
  # Precompute neighbor indices for each cell-year
  neighbor_lookup <- vector("list", length = nrow(data))
  
  for (cell_idx in seq_along(id_order)) {
    cell_neighbors <- neighbors[[cell_idx]]
    if (length(cell_neighbors) == 0) next
    for (year_idx in seq_along(years)) {
      row_id <- row_index_matrix[cell_idx, year_idx]
      if (is.na(row_id)) next
      neighbor_rows <- row_index_matrix[cell_neighbors, year_idx]
      neighbor_lookup[[row_id]] <- neighbor_rows[!is.na(neighbor_rows)]
    }
  }
  
  # Compute neighbor stats for each variable
  for (var_name in vars) {
    vals <- data[[var_name]]
    stats_matrix <- matrix(NA_real_, nrow = nrow(data), ncol = 3)
    for (i in seq_along(neighbor_lookup)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          stats_matrix[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    colnames(stats_matrix) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    data <- cbind(data, stats_matrix)
  }
  
  data
}

# Usage:
years <- 1992:2019
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, years, neighbor_source_vars)
```

---

### **Why This Is Faster**
- Eliminates string concatenation and hash lookups.
- Uses numeric indexing and precomputed matrices.
- Builds `neighbor_lookup` **once** for all years and variables.
- Memory-efficient: `neighbor_lookup` is a list of integer vectors, not strings.

---

**Expected Performance Gain**  
From 86+ hours â†’ likely **minutes to a few hours**, as the bottleneck (string operations and repeated hashing) is removed and replaced with integer-based lookups.

**Preserves**:
- Original Random Forest model.
- Original estimand and feature definitions.
END_RESPONSE>>>
