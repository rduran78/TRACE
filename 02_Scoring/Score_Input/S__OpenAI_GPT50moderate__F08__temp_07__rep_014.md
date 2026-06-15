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
run_id: OpenAI_F08_T07_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The extreme runtime (86+ hours) is caused by repeatedly computing neighbor stats for every cell-year row across 6.46M rows. The neighbor relationships are static across years, but the current design recalculates neighbor indices for each row/year. This creates massive redundant work and memory overhead. The neighbor lookup should be computed **once per cell** (344K entries), not per cell-year row. Then, for each year, compute neighbor stats by slicing the yearâ€™s values and applying precomputed neighbor indices.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** for the 344,208 cells using their IDs (static graph).
2. Split the panel data by year and compute neighbor stats year-by-year using the static neighbor mapping.
3. Use `matrix` operations or `data.table` for efficiency.
4. Append results back to the original `cell_data` without breaking the Random Forest pipeline.
5. Avoid growing objects in loops; preallocate or use `rbindlist`.

---

**Optimized R Code**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup (static per cell)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) as.integer(id_order[neighbors[[i]]]))
}
neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
names(neighbor_lookup_static) <- as.character(id_order)

# Function to compute neighbor stats for a given year
compute_neighbor_stats_year <- function(dt_year, var_name, neighbor_lookup) {
  vals <- dt_year[[var_name]]
  res <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
  
  for (i in seq_len(nrow(dt_year))) {
    neigh_ids <- neighbor_lookup[[ as.character(dt_year$id[i]) ]]
    if (length(neigh_ids) == 0) next
    neighbor_vals <- vals[ match(neigh_ids, dt_year$id, nomatch = 0) ]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    res[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  res
}

# Main optimized loop: process by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {
  # Preallocate storage for all rows
  all_stats <- vector("list", length(years))
  
  for (y_idx in seq_along(years)) {
    y <- years[y_idx]
    dt_year <- cell_data[year == y]
    stats_year <- compute_neighbor_stats_year(dt_year, var_name, neighbor_lookup_static)
    all_stats[[y_idx]] <- stats_year
  }
  
  # Bind results and add to cell_data in correct order
  all_stats_mat <- do.call(rbind, all_stats)
  
  cell_data[, paste0(var_name, "_nbr_max") := all_stats_mat[,1]]
  cell_data[, paste0(var_name, "_nbr_min") := all_stats_mat[,2]]
  cell_data[, paste0(var_name, "_nbr_mean") := all_stats_mat[,3]]
}

# At this point, cell_data has neighbor stats ready for prediction
# Preserve trained Random Forest model and use it as before
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

**Key Gains**  
- **Neighbor graph built once** (no per-row repetition).
- **Year-level batch processing** reduces complexity from O(N*T*k) to O(T*(N+k)).
- Should reduce runtime from 86+ hours to a few hours (or less) on 16 GB RAM.

**Numerical estimand preserved**: Same neighbor stats, same Random Forest model, only optimized computation.
END_RESPONSE>>>
