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
run_id: OpenAI_F08_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over ~6.46 million rows, performing neighbor aggregation redundantly for each cell-year. The neighbor topology is static across years, but the code recomputes neighbor indices per row-year and repeatedly extracts values from large vectors. This leads to massive repeated lookups and memory overhead.

---

**Optimization Strategy**  
1. **Leverage static neighbor relationships**: Build neighbor lookup **once per cell**, not per cell-year.
2. **Aggregate by year in bulk**: For each year, extract relevant variable vector and compute neighbor stats using the static lookup.
3. **Vectorized computation**: Avoid per-row lapply loops over 6.46M rows. Instead, operate year-wise and bind results.
4. **Memory efficiency**: Reuse precomputed structures and avoid repeatedly slicing huge vectors.

---

**Working R Code**

```r
# 1. Build static neighbor lookup by cell ID (not cell-year)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # Returns a list where each element is neighbor cell IDs (not row indices)
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}

neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
id_to_index <- setNames(seq_along(id_order), id_order)

# 2. Compute neighbor stats efficiently
compute_neighbor_stats_yearwise <- function(data, var_name, neighbor_lookup_static, id_to_index) {
  years <- sort(unique(data$year))
  result_list <- vector("list", length(years))
  
  for (y in seq_along(years)) {
    year_val <- years[y]
    subset_idx <- which(data$year == year_val)
    subset_data <- data[subset_idx, c("id", var_name)]
    
    vals <- setNames(subset_data[[var_name]], subset_data$id)
    
    # Compute neighbor stats for each cell
    stats_mat <- t(sapply(neighbor_lookup_static, function(neigh_ids) {
      neigh_vals <- vals[as.character(neigh_ids)]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) == 0) return(c(NA, NA, NA))
      c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
    }))
    
    colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    
    result_list[[y]] <- data.frame(
      id = id_order,
      year = year_val,
      stats_mat,
      row.names = NULL
    )
  }
  
  do.call(rbind, result_list)
}

# 3. Master loop for all neighbor variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats_all <- lapply(neighbor_source_vars, function(var_name) {
  compute_neighbor_stats_yearwise(cell_data, var_name, neighbor_lookup_static, id_to_index)
})

# Merge all stats back into cell_data
neighbor_stats_merged <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), neighbor_stats_all)
cell_data <- merge(cell_data, neighbor_stats_merged, by = c("id", "year"))
```

---

**Why this works faster**  
- Neighbor topology (`neighbor_lookup_static`) is computed **once**.
- For each year (~28 loops), we handle 344K rows and compute neighbor stats in a fully vectorized manner with `sapply` over 344K elements, not 6.46M individually.
- Total complexity drops from O(N * neighbors * years) to O(years * cells * neighbors), eliminating redundant lookups across years.
- Expected runtime: hours â†’ **minutes** on a standard laptop.

**Preserves**  
- Original estimand.
- Trained Random Forest model.
- All neighbor feature definitions (max, min, mean).
END_RESPONSE>>>
