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
run_id: OpenAI_F07_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck comes from repeatedly performing expensive list operations and copying during neighbor lookups and aggregations for ~6.46M rows. Specifically:  
- `lapply` per row with string concatenation and repeated vector lookups is O(N * k) with high overhead.  
- The neighbor relationships are static across years, but code rebuilds keys per row-year.  
- Lack of vectorization and precomputation: repeated joins instead of a single indexed merge.  

**Optimization Strategy**  
1. **Precompute neighbor relationships at the cell level** and reuse across years instead of reconstructing per row-year.  
2. **Explode to (row, neighbor) pairs once** as an indexed table, then join values in a data.table pipeline for aggregation.  
3. Use **data.table** for fast group operations and memory efficiency.  
4. Compute all five variables in one pass rather than looping variable by variable.  
5. Avoid string concatenation; use integer keys for joins.  

This reduces complexity from millions of small lapply operations to a few large vectorized joins and aggregations.

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in adjacency order
# rook_neighbors_unique: list of integer neighbor indices (spdep::nb)

setDT(cell_data)
setkey(cell_data, id, year)

# 1. Precompute neighbor pairs at cell level
cell_ids <- id_order
nb_pairs <- data.table(
  src = rep(seq_along(cell_ids), lengths(rook_neighbors_unique)),
  nbr = unlist(rook_neighbors_unique)
)
nb_pairs[, src_id := cell_ids[src]]
nb_pairs[, nbr_id := cell_ids[nbr]]
nb_pairs[, c("src", "nbr") := NULL]

# 2. Expand to panel by joining on year
years <- unique(cell_data$year)
panel_pairs <- nb_pairs[, .(id = src_id, neighbor_id = nbr_id)][
  CJ(year = years, id = src_id, unique = TRUE), on = .(id)
][, .(id, year, neighbor_id)]

# 3. Join neighbor values
panel_pairs <- panel_pairs[cell_data, on = .(neighbor_id = id, year),
  nomatch = 0,
  allow.cartesian = TRUE
]

# panel_pairs now has columns: id (source), year, neighbor_id, ntl, ec, pop_density, def, usd_est_n2

# 4. Aggregate neighbor stats
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_exprs <- lapply(neighbor_vars, function(v) {
  list(
    max = as.call(list(as.name("max"), as.name(v), na.rm = TRUE)),
    min = as.call(list(as.name("min"), as.name(v), na.rm = TRUE)),
    mean = as.call(list(as.name("mean"), as.name(v), na.rm = TRUE))
  )
})

agg_list <- unlist(lapply(seq_along(neighbor_vars), function(i) {
  v <- neighbor_vars[i]
  c(
    setNames(list(max(get(v), na.rm = TRUE)), paste0(v, "_nbr_max")),
    setNames(list(min(get(v), na.rm = TRUE)), paste0(v, "_nbr_min")),
    setNames(list(mean(get(v), na.rm = TRUE)), paste0(v, "_nbr_mean"))
  )
}), recursive = FALSE)

neighbor_stats <- panel_pairs[, agg_list, by = .(id, year)]

# 5. Merge back to cell_data
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# Result: cell_data now has neighbor max, min, mean features for each variable
```

---

### **Expected Performance Improvement**
- Single join and grouped aggregation replaces millions of lapply calls.
- Scales to ~6.5M rows on 16 GB RAM using data.table.
- Expected runtime: **minutes to a few hours** instead of 86+ hours.

**Preserves:**  
- Original rook adjacency.
- Panel structure.
- Numerical estimand.
- Random Forest model remains untouched.
END_RESPONSE>>>
