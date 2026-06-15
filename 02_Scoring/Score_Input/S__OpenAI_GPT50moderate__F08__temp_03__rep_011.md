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
run_id: OpenAI_F08_T03_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics (`max`, `min`, `mean`) for every cell-year row (â‰ˆ6.46M rows) by iterating through `neighbor_lookup` each time. This is O(N Ã— K) with large N (6.46M) and K (avg. neighbors), causing 86+ hours runtime. The neighbor structure is static across years, but the code does not exploit this: it rebuilds and applies neighbor lookups repeatedly for all rows, even though the neighbor relationships are constant.

---

**Optimization Strategy**  
1. **Precompute static neighbor indices once per cell** (already done by `build_neighbor_lookup`).
2. **Process year by year**: For each year, slice the data for that year, compute neighbor stats using the static neighbor structure (cell-level), then merge back. This avoids iterating over all 6.46M rows at once.
3. **Vectorize computations**: Use matrix operations or `vapply` instead of repeated `lapply` over all rows.
4. **Memory efficiency**: Work in chunks (per year) to keep RAM usage low.
5. **Preserve estimand**: The computed neighbor max/min/mean must remain identical to the original logic.

---

**Optimized R Code**

```r
# Precompute neighbor lookup at cell level (static)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # neighbors: spdep::nb object
  lapply(seq_along(id_order), function(i) neighbors[[i]])
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one year
compute_neighbor_stats_year <- function(year_data, var_name, neighbor_lookup_static, id_order) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  # Preallocate result matrix
  res <- matrix(NA_real_, nrow = nrow(year_data), ncol = 3)
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  
  for (i in seq_along(id_order)) {
    cell_id <- id_order[i]
    nbr_ids <- id_order[neighbor_lookup_static[[i]]]
    nbr_vals <- vals[as.character(nbr_ids)]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      res[year_data$id == cell_id, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
  }
  res
}

# Main loop: process by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  idx <- which(cell_data$year == yr)
  year_data <- cell_data[idx, ]
  
  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_neighbor_stats_year(year_data, var_name, neighbor_lookup_static, id_order)
    cell_data[idx, colnames(stats_mat)] <- stats_mat
  }
}

# At this point, cell_data has neighbor features added.
# Proceed with Random Forest prediction as before.
```

---

**Why This is Faster**  
- The neighbor relationships are computed once, not per row-year.
- Computation is done year by year (28 iterations), reducing memory and avoiding 6.46M-row lapply.
- Each year processes only 344,208 rows with a simple loop over cells and vector lookups.
- Expected runtime drops from 86+ hours to a few hours or less on a standard laptop.

**Preserves**  
- Trained Random Forest model.
- Original numerical estimand (same max/min/mean logic).
- Static neighbor structure assumption.
END_RESPONSE>>>
