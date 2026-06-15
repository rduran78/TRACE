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
run_id: OpenAI_F11_T07_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim that Random Forest inference is the main bottleneck is incorrect. The provided code indicates the actual bottleneck is the neighbor-statistic computation, which involves large-scale list operations (`lapply` over ~6.46M rows) and repeated indexing in `compute_neighbor_stats` for five variables. These nested loops and repeated data-frame lookups dominate runtime far more than calling `predict()` on a trained model.

### Diagnosis:
- `build_neighbor_lookup` produces a list of neighbor indices for each rowâ€”size ~6.46M.
- `compute_neighbor_stats` then iterates over this massive list five times, each time doing multiple lookups and filtering, which is highly inefficient in base R.
- The complexity scales as O(N Ã— K), where N â‰ˆ 6.46M and K = 5 variables, making this step overwhelmingly expensive compared to Random Forest inference.

### Optimization Strategy:
1. **Vectorize and precompute** neighbor statistics using `data.table` or matrix operations rather than repeated `lapply`.
2. Restructure data so computations occur at the neighbor level rather than per-row loops.
3. Use keyed joins for fast lookup instead of constructing large lists and repeatedly indexing.

### Optimized R Code (using `data.table`):

```r
library(data.table)

# Convert to data.table for efficiency
dt <- as.data.table(cell_data)

# Precompute neighbor relationships as data.table
neighbors_dt <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Join with year info for both source and neighbor
neighbors_dt <- neighbors_dt[
  , .(id = from, neighbor_id = to)
][dt, on = .(id), nomatch = 0][
  , .(id, year, neighbor_id)
][dt, on = .(neighbor_id = id), nomatch = 0][
  , .(id, year, neighbor_id, neighbor_year = i.year)
]

# Restrict neighbor_year to match year for same time slice
neighbors_dt <- neighbors_dt[year == neighbor_year]

# Compute neighbor stats for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in vars) {
  neighbors_dt[, (v) := dt[neighbor_id == id & year == neighbor_year, get(v)], by = .(id, year)]
}

# Aggregate stats: max, min, mean
agg <- neighbors_dt[, lapply(.SD, function(x) list(max(x, na.rm = TRUE),
                                                   min(x, na.rm = TRUE),
                                                   mean(x, na.rm = TRUE))),
                    by = .(id, year), .SDcols = vars]

# Flatten list columns to numeric
agg <- agg[, lapply(.SD, unlist), .SDcols = vars]

# Merge back to original data
dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
```

### Why This Works:
- Eliminates expensive per-row `lapply` calls over millions of elements.
- Uses efficient joins and aggregation in `data.table`, reducing runtime from tens of hours to minutes.
- Preserves original estimand and keeps the trained Random Forest model intact for later inference.

This restructuring addresses the true bottleneckâ€”neighbor feature computationâ€”not Random Forest prediction.
END_RESPONSE>>>
