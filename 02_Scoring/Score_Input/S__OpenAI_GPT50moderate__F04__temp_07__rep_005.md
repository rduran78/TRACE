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
run_id: OpenAI_F04_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is indeed the neighbor feature construction, not Random Forest inference. The current implementation repeatedly uses `lapply` over ~6.46M rows, building neighbor indices and computing statistics in pure R loops. This is extremely inefficient because:

- `build_neighbor_lookup` constructs a large nested list of integer vectors via R loops.
- `compute_neighbor_stats` iterates again through all rows for each variable (5 times), performing repeated lookups and aggregations.
- Memory footprint and function call overhead dominate performance.

Given 6.46M rows and 5 variables, this approach does tens of millions of small R operations.  

---

**Optimization Strategy**  
1. **Vectorize neighbor relationships**: Flatten neighbor pairs into a long data frame (cell-year row â†’ neighbor row) once.
2. **Use `data.table` for aggregation**: Compute max/min/mean per focal row across neighbors efficiently in C-optimized code.
3. **Reuse join structure**: Instead of recalculating per variable, join values in one pass and compute summaries grouped by focal row.
4. **Parallelize aggregation**: Use `data.table` multithreading (built-in) or `future.apply` if necessary.
5. **Avoid storing huge lists**: Work with integer vectors and keyed joins instead.

This converts the problem into a large but efficient grouped aggregation task.

---

**Optimized R Code using `data.table`**  
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor indices for each cell (by id_order)

# Convert cell_data to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Step 1: Build long neighbor table (cell-year to neighbor cell-year)
id_order <- unique(cell_data$id)
id_to_idx <- setNames(seq_along(id_order), id_order)

# Precompute all neighbor pairs (cell -> neighbors)
pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand pairs across years (cartesian join)
years <- sort(unique(cell_data$year))
pairs_year <- pairs[, .(id, neighbor_id), keyby = .(id)][
  , .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Map to row indices
setkey(cell_data, id, year)
pairs_year[, focal_idx := cell_data[.(id, year), which = TRUE]]
pairs_year[, neigh_idx := cell_data[.(neighbor_id, year), which = TRUE]]

# Drop missing
pairs_year <- pairs_year[!is.na(focal_idx) & !is.na(neigh_idx)]

# Step 2: Compute neighbor stats for all vars in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  pairs_year[, value := vals[neigh_idx]]
  
  agg <- pairs_year[!is.na(value),
    .(max_val = max(value), min_val = min(value), mean_val = mean(value)),
    by = focal_idx]
  
  # Add back to main table
  cell_data[agg$focal_idx,
            paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")) :=
              .(agg$max_val, agg$min_val, agg$mean_val)]
}

# cell_data now has neighbor features for all variables
```

---

**Why this is faster**  
- `data.table` aggregation operates in C with multi-threading.
- We avoid building huge nested lists.
- Only one neighbor mapping pass; then reuse for all variables.
- Memory-efficient: works with integer indices and vectorized joins.

---

**Expected result**  
- Runtime reduced from 86+ hours to a few hours (or less) on a 16 GB machine.
- Preserves the original numerical estimand and Random Forest model.
- Scalable to millions of rows and large neighbor sets.
END_RESPONSE>>>
