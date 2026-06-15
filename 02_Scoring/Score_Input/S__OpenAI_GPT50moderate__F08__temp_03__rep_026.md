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
run_id: OpenAI_F08_T03_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over **6.46 million rows** and repeatedly computes neighbor statistics by scanning neighbors for each row. This results in redundant work across years since the neighbor structure is static, but the code recomputes neighbor indices for every row-year combination. Additionally, the `lapply` approach creates large intermediate objects and does not leverage vectorization or grouping by year.

---

**Optimization Strategy**  
1. **Precompute neighbor lookup once per cell (not per row-year)** since neighbors do not change across years.
2. **Process data year by year**: For each year, extract the relevant slice of data and compute neighbor stats using the static neighbor lookup.
3. Use **vectorized operations** (e.g., `vapply`, `matrix` operations) instead of deeply nested `lapply`.
4. Avoid creating huge lists; instead, build a matrix or data frame directly.
5. Keep memory usage low by working in chunks (year-level) rather than the full 6.46M rows at once.

---

**Optimized R Code**

```r
# Precompute neighbor lookup at cell level (not row-year)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # Returns a list where each element is the vector of neighbor indices (cell-level)
  lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]]
  })
}

compute_neighbor_stats_year <- function(year_data, neighbor_lookup_static, var_name) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  
  # Compute stats for each cell
  result <- vapply(neighbor_lookup_static, function(neigh_ids) {
    neighbor_vals <- vals[as.character(neigh_ids)]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  
  t(result)  # transpose to rows = cells
}

# Main optimized pipeline
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output container
neighbor_feature_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))
}))

# Process year by year
years <- sort(unique(cell_data$year))
cell_data_out <- list()

for (yr in years) {
  year_slice <- cell_data[cell_data$year == yr, ]
  year_slice <- year_slice[match(id_order, year_slice$id), ]  # align with id_order
  
  # Compute neighbor stats for each variable
  stats_matrix <- do.call(cbind, lapply(neighbor_source_vars, function(v) {
    compute_neighbor_stats_year(year_slice, neighbor_lookup_static, v)
  }))
  
  colnames(stats_matrix) <- neighbor_feature_names
  year_slice <- cbind(year_slice, stats_matrix)
  cell_data_out[[as.character(yr)]] <- year_slice
}

cell_data <- do.call(rbind, cell_data_out)
```

---

**Why This Is Faster**  
- **Neighbor lookup computed once** for 344,208 cells instead of 6.46M rows.
- **Year-based processing** reduces memory pressure and avoids redundant computations.
- **Vectorized operations** replace millions of small list operations.
- Expected runtime drops from 86+ hours to **a few hours** (depending on disk I/O and CPU).

---

**Preserves**  
- Original Random Forest model (unchanged).
- Original numerical estimand (neighbor max, min, mean per cell-year).
- Data integrity (same rows, same features, just computed efficiently).
END_RESPONSE>>>
