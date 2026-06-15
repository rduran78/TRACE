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
run_id: OpenAI_F04_T07_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is the repeated R `lapply` loops and string concatenation in `build_neighbor_lookup` and `compute_neighbor_stats`. For 6.46 million rows and 5 variables, these operations create enormous overhead in memory allocation and interpretation. The neighbor lookup is recomputed row-wise and then stats are computed via nested loops, which is inefficient for large panel data.

**Optimization Strategy**  
- Precompute neighbor indices as integer vectors without repeated string concatenation.
- Use `data.table` for fast keyed joins instead of constructing keys manually.
- Vectorize neighbor stats computation using aggregation rather than looping.
- Avoid repeated object copies; compute all neighbor features in a single pass.

**Optimized R Code**

```r
library(data.table)

# Convert to data.table for fast operations
setDT(cell_data)

# Precompute lookup table for (id, year) to row index
cell_data[, key := .I]
setkey(cell_data, id, year)

# Flatten neighbor relationships across all cells
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(ref_idx) {
    data.table(
      id = id_order[ref_idx],
      neighbor_id = id_order[rook_neighbors_unique[[ref_idx]]]
    )
  })
)

# Expand to all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, each = length(years)),
                                neighbor_id = rep(neighbor_id, each = length(years)),
                                year = rep(years, times = .N))]

# Join with cell_data to get row indices
neighbor_dt[, id_key := cell_data[J(id, year), key]]
neighbor_dt[, neighbor_key := cell_data[J(neighbor_id, year), key]]

# Remove NAs (missing neighbor-year combos)
neighbor_dt <- neighbor_dt[!is.na(neighbor_key)]

# Compute neighbor stats in one grouped pass
compute_neighbor_features <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, val := vals[neighbor_key]]
  neighbor_dt[!is.na(val),
    .(max_val = max(val),
      min_val = min(val),
      mean_val = mean(val)),
    by = id_key
  ]
}

# Apply for all variables and merge back
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  stats <- compute_neighbor_features(var_name)
  setnames(stats, c("id_key", paste0(var_name, "_nb_max"),
                    paste0(var_name, "_nb_min"),
                    paste0(var_name, "_nb_mean")))
  cell_data[stats, on = .(key = id_key)] <- stats[, -1]
}
```

**Why Faster?**  
- Eliminates repeated string operations.
- Uses `data.table` joins instead of costly per-row lookups.
- Vectorized aggregation reduces R interpreter overhead.
- Handles all years and all variables in efficient batch steps.

**Expected Performance Gain**  
From 86+ hours to a few hours on a 16 GB machine due to reduced complexity and optimized memory access. Model remains intact, numerical estimand preserved.
END_RESPONSE>>>
