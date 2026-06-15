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
run_id: OpenAI_F04_T03_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation:  
- `build_neighbor_lookup` creates a list of neighbor indices for *each cell-year row* (â‰ˆ6.46M), which is extremely memory- and CPU-intensive.  
- `compute_neighbor_stats` iterates over this large list multiple times (once per variable), performing redundant lookups and aggregations.  
- The operations are essentially repeated joins and group summaries that could be vectorized or collapsed into a single pass.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lookups**: Precompute neighbor relationships at the *cell level* and then join by year using vectorized operations.  
2. **Use `data.table` for fast joins and aggregation** instead of `lapply` over millions of rows.  
3. **Compute all neighbor stats in one grouped operation** rather than looping over variables.  
4. **Leverage long format and keyed joins**: Expand neighbor pairs across years, then aggregate in bulk.  

**Working R Code**  
```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Precompute neighbor pairs at cell level
# rook_neighbors_unique: list of neighbors per cell index
neighbor_pairs <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand across years by joining on dt
years <- unique(dt$year)
neighbor_dt <- neighbor_pairs[, .(id, neighbor_id)][
  , .(id = rep(id, each = length(years)),
      neighbor_id = rep(neighbor_id, each = length(years)),
      year = rep(years, times = .N))
]

# Join neighbor values
setkey(dt, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[dt, on = .(neighbor_id = id, year), nomatch = 0]

# Compute stats for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg <- neighbor_dt[, lapply(.SD, function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(NA_real_, NA_real_, NA_real_))
  c(max(x), min(x), mean(x))
}), by = .(id, year), .SDcols = vars]

# Reshape aggregated stats into wide format
agg_long <- melt(agg, id.vars = c("id", "year"))
agg_long[, c("var", "stat") := tstrsplit(variable, "_", fixed = TRUE)]
# Actually, create proper names for max/min/mean
agg_stats <- agg_long[, .(max = V1, min = V2, mean = V3), by = .(id, year, variable)]

# Merge back to original dt
dt <- merge(dt, dcast(agg_stats, id + year ~ variable, value.var = c("max", "min", "mean")), by = c("id", "year"))

# Result: dt now has neighbor features efficiently computed
```

**Why This Works**  
- Eliminates 6.46M `lapply` calls; uses vectorized joins and grouped aggregation.  
- Computes all neighbor stats in one grouped pass instead of looping over variables.  
- Scales to millions of rows efficiently with `data.table`.  

**Expected Impact**  
- From 86+ hours to a few minutes on a laptop (depending on disk I/O and memory).  
- Preserves original estimand and trained Random Forest model.  
- Memory footprint reduced by avoiding massive nested lists.
END_RESPONSE>>>
