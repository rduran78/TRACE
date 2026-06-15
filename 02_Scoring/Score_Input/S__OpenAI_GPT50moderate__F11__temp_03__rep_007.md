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
run_id: OpenAI_F11_T03_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence shows the main bottleneck is **neighbor feature computation**, not Random Forest inference. The `build_neighbor_lookup` and `compute_neighbor_stats` functions perform millions of `lapply` iterations and repeated vector lookups across 6.46M rows Ã— 5 variables, which is extremely slow in R. Random Forest `predict()` on 6.46M rows with 110 variables is heavy but typically runs in minutes to a few hours, not 86+ hours.

### Diagnosis
- `build_neighbor_lookup` creates a large list of length 6.46M with nested lookups.
- `compute_neighbor_stats` repeatedly scans and subsets vectors inside `lapply`.
- These operations are pure R loops with high overhead, dominating runtime.

### Optimization Strategy
- Precompute neighbor indices once in an efficient structure.
- Use **vectorized joins** or **data.table** grouping instead of per-row `lapply`.
- Compute neighbor stats in bulk using joins keyed by `(id, year)` pairs.
- Preserve the Random Forest model and estimand.

### Optimized R Code (data.table)

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create a lookup table of neighbors expanded by year
id_year <- CJ(id = id_order, year = unique(cell_data$year))  # all id-year combos
neighbors_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)
neighbors_dt <- neighbors_dt[id_year, on = .(id), allow.cartesian = TRUE]
neighbors_dt[, year := i.year]  # propagate year
neighbors_dt[, i.year := NULL]

# Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in vars) {
  tmp <- cell_data[, .(id, year, val = get(v))]
  setnames(tmp, "val", v)
  neighbors_dt <- neighbors_dt[tmp, on = .(neighbor_id = id, year)]
}

# Compute stats by (id, year)
stat_cols <- function(x) list(max = max(x, na.rm = TRUE),
                              min = min(x, na.rm = TRUE),
                              mean = mean(x, na.rm = TRUE))

agg_list <- c(lapply(vars, function(v) substitute(stat_cols(x), list(x = as.name(v)))))
names(agg_list) <- vars

neighbor_stats <- neighbors_dt[, c(.(id, year), lapply(.SD, function(x) {
  list(max = max(x, na.rm = TRUE),
       min = min(x, na.rm = TRUE),
       mean = mean(x, na.rm = TRUE))
})), by = .(id, year), .SDcols = vars]

# Merge back to cell_data
cell_data <- neighbor_stats[cell_data, on = .(id, year)]
```

### Why This Works
- Eliminates per-row `lapply` loops.
- Uses `data.table` joins and group aggregation, which are highly optimized in C.
- Handles all neighbor variables in one pass.
- Preserves original estimand and Random Forest model.

**Expected runtime improvement:** From 86+ hours to under 1â€“2 hours on a 16 GB laptop.
END_RESPONSE>>>
