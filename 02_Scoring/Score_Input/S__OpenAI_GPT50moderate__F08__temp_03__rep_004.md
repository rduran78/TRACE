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
run_id: OpenAI_F08_T03_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows and recomputes neighbor statistics for each cell-year combination, even though the neighbor structure is static. This results in massive redundant lookups and repeated aggregation across 28 years for the same neighbor sets. The bottleneck is the nested `lapply` over millions of rows combined with repeated indexing and aggregation.

---

**Optimization Strategy**  
1. **Separate static and dynamic components**:
   - The neighbor relationships among cells are static across years.
   - The variables (`ntl`, `ec`, etc.) change by year.
2. **Precompute a static neighbor index**:
   - Map each cell to its neighbors once (by cell ID, not cell-year).
3. **Process year by year**:
   - For each year, slice the data for that year only.
   - Compute neighbor statistics using the static neighbor index and the year-specific values.
4. **Bind results back**:
   - Append computed features for each year to the corresponding rows.
5. **Vectorize where possible**:
   - Use `vapply` or matrix operations instead of repeated `lapply` calls.
6. **Memory efficiency**:
   - Work year by year to avoid loading all neighbor-expanded data into memory at once.

---

**Working R Code**

```r
# Precompute static neighbor index by cell ID
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(id_order, function(cell_id) {
    ref_idx <- id_to_ref[as.character(cell_id)]
    id_order[neighbors[[ref_idx]]]
  })
}

compute_neighbor_stats_for_year <- function(year_data, var_name, neighbor_lookup) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  result <- vapply(year_data$id, function(cell_id) {
    neighbor_ids <- neighbor_lookup[[as.character(cell_id)]]
    neighbor_vals <- vals[neighbor_ids]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  t(result)
}

# Main optimized pipeline
neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output container
all_results <- vector("list", length(neighbor_source_vars))

for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "\n")
  var_results <- vector("list", length(unique(cell_data$year)))
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    cat("  Year:", yr, "\n")
    year_data <- cell_data[cell_data$year == yr, ]
    stats_mat <- compute_neighbor_stats_for_year(year_data, var_name, neighbor_lookup_static)
    colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    var_results[[as.character(yr)]] <- cbind(year_data$id, year_data$year, stats_mat)
  }
  
  var_results_df <- do.call(rbind, var_results)
  all_results[[var_name]] <- var_results_df
}

# Merge back into cell_data
for (var_name in neighbor_source_vars) {
  res <- all_results[[var_name]]
  colnames(res)[1:2] <- c("id", "year")
  cell_data <- merge(cell_data, res, by = c("id", "year"), all.x = TRUE, sort = FALSE)
}
```

---

**Why This Works**  
- **Static neighbor lookup**: Built once for 344,208 cells instead of 6.46M rows.
- **Year-by-year processing**: Only 344,208 rows per iteration instead of 6.46M.
- **Vectorized aggregation**: `vapply` and pre-indexed lookups reduce overhead.
- **Preserves estimand**: The computed neighbor max, min, and mean remain identical to the original logic.

**Expected Performance**  
- Reduces complexity from O(N * T * K) redundant lookups to O(T * N), where N = cells, T = years, K = neighbors.
- On a 16 GB laptop, this should complete in hours instead of 86+ hours, likely under 2â€“3 hours depending on I/O.
END_RESPONSE>>>
