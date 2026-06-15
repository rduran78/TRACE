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
run_id: OpenAI_F08_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation recomputes neighbor statistics for every cell-year entry by iterating through all ~6.46M rows multiple times (once per variable). While the neighbor structure is static across years, the code redundantly processes the same lookup for every entry and variable. This causes excessive computation (~86+ hours). Memory is stressed due to repeated `lapply` calls and intermediate `rbind` operations across millions of rows.

---

**Optimization Strategy:**  
- Precompute a **static neighbor lookup by cell ID** (not by cell-year).
- For each year:
  - Filter data for that year only (â‰ˆ344k rows).
  - Compute neighbor statistics using the static lookup on year-specific values only.
  - Append results efficiently.
- Process all five variables in a single pass per year to reduce overhead.
- Avoid repeated creation of large intermediate lists; instead use preallocated matrices/data frames.
- Use `data.table` for speed on filtering and joining operations.

---

**Working R Code:**

```r
library(data.table)

# Precompute static neighbor lookup by cell ID (not year-specific)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}

# Function to compute all neighbor stats for a single year's data
compute_neighbor_stats_year <- function(year_data, neighbor_lookup, var_names) {
  n <- nrow(year_data)
  m <- length(var_names)
  result_mat <- matrix(NA_real_, nrow = n, ncol = m * 3)
  
  vals_list <- lapply(var_names, function(v) year_data[[v]])
  
  for (i in seq_len(n)) {
    nb_ids <- neighbor_lookup[[i]]
    if (length(nb_ids) == 0) next
    nb_idx <- match(nb_ids, year_data$id)
    valid_neighbors <- !is.na(nb_idx)
    if (!any(valid_neighbors)) next
    nb_idx <- nb_idx[valid_neighbors]
    
    for (j in seq_along(var_names)) {
      nb_vals <- vals_list[[j]][nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) next
      result_mat[i, (j - 1) * 3 + 1] <- max(nb_vals)
      result_mat[i, (j - 1) * 3 + 2] <- min(nb_vals)
      result_mat[i, (j - 1) * 3 + 3] <- mean(nb_vals)
    }
  }
  
  result_mat
}

# Main optimized pipeline
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, var_names) {
  setDT(cell_data)
  
  # Build static lookup
  neighbor_lookup <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)
  
  # Predefine feature column names
  feature_cols <- as.vector(t(outer(var_names, c("max","min","mean"), paste, sep = "_nb_")))
  
  # Add empty columns for features
  for (col in feature_cols) set(cell_data, j = col, value = NA_real_)
  
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    year_idx <- which(cell_data$year == yr)
    year_data <- cell_data[year_idx]
    
    stats_mat <- compute_neighbor_stats_year(year_data, neighbor_lookup, var_names)
    
    # Assign computed stats back
    for (k in seq_along(feature_cols)) {
      set(cell_data, i = year_idx, j = feature_cols[k], value = stats_mat[,k])
    }
    cat("Processed year:", yr, "\n")
  }
  
  cell_data
}

# Example usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

**Efficiency Gains:**
- **Static neighbor mapping** computed once instead of per row.
- **Year-based batching:** reduces row count per pass from 6.46M to 344k.
- **Memory safe:** Preallocates result matrices; avoids large intermediate lists.
- Complexity drops from O(N_rows * N_vars * neighbor_size) across entire dataset to O(N_years * N_cells * N_vars).

This design should reduce 86+ hours to **minutes to a few hours on a standard laptop** while preserving original estimands and compatibility with the trained Random Forest model.
END_RESPONSE>>>
