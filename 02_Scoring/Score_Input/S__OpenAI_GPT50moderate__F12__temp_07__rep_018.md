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
run_id: OpenAI_F12_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is **incorrect**. The main bottleneck is **not** `do.call(rbind, result)` or repeated list-binding in `compute_neighbor_stats()`. Evidence:

- `result` is an `lapply` of length â‰ˆ **6.46M** (one per cell-year row). `do.call(rbind, â€¦)` on 6.46M small numeric vectors (length 3) adds overhead, but the real cost lies in the **inner loop**: repeatedly subsetting `vals[idx]` for each observation.
- Each `neighbor_lookup[[i]]` is a small integer vector, but there are **6.46M lookups**, each performing an R-level loop and multiple allocations.
- This is repeated **5 times** for 5 variables â†’ ~32M neighbor aggregation operations. This dominates runtime far more than a single `rbind` call.

### Correct Diagnosis
The hotspot is the **per-row neighbor aggregation implemented in R loops**. The vectorized `do.call(rbind, â€¦)` is negligible compared to millions of `vals[idx]` subsettings and repeated filtering of `NA`s.

---

### Optimization Strategy
- Use **data.table** or **matrix aggregation** to compute all neighbor stats in one or few grouped operations instead of looping 6.46M times.
- Pre-flatten neighbor relationships into a long table `(row_id, neighbor_id)` and join once.
- Compute `max`, `min`, and `mean` by group using fast C-level aggregation.
- Avoid repeated passes for each variableâ€”handle all 5 variables in one grouped computation.

---

### Optimized R Code

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute neighbor lookup as pairs (cell_id, neighbor_id)
# id_order aligns rook_neighbors_unique indices with actual IDs
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years
years <- unique(dt$year)
neighbor_dt <- neighbor_pairs[CJ(id = id, year = years), on = "id", allow.cartesian = TRUE]
neighbor_dt[, neighbor_year := year]
# Join neighbor values
setkey(neighbor_dt, neighbor_id, neighbor_year)
setkey(dt, id, year)

vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in vars) {
  neighbor_dt[, (v) := dt[.(neighbor_id, neighbor_year), on = .(id, year), get(v)]]
}

# Compute stats by (id, year) group
agg <- neighbor_dt[, {
  lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) c(NA, NA, NA) else c(max(x), min(x), mean(x))
  })
}, by = .(id, year), .SDcols = vars]

# Reshape and merge back
stat_names <- c("max", "min", "mean")
new_cols <- unlist(lapply(vars, function(v) paste0(v, "_neighbor_", stat_names)))
agg <- cbind(agg[, .(id, year)], setDT(do.call(cbind, agg[, -c("id","year")])))

setnames(agg, c("id","year", new_cols))
setkey(agg, id, year)
dt <- agg[dt]  # merge back
```

---

### Why This Fix Works
- Eliminates **6.46M R-level loops**; instead uses **vectorized joins and group aggregation in C**.
- Handles all variables in one pass.
- Memory-efficient: works on long format neighbor table (~1.3M pairs Ã— 28 years â‰ˆ 38M rows), which is feasible on 16 GB RAM with `data.table`.
- Preserves original estimand and trained model.

**Expected speed-up:** From ~86+ hours to **minutes**.
END_RESPONSE>>>
