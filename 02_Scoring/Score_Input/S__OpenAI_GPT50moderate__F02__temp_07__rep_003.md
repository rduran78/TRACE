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
run_id: OpenAI_F02_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottlenecks are:  
- `build_neighbor_lookup` creates a massive list (â‰ˆ6.46M elements) using `lapply`, consuming huge memory.  
- `compute_neighbor_stats` iterates through these lists repeatedly for each variable, causing redundant lookups.  
- Multiple passes over 6.46M rows Ã— 5 variables = enormous overhead.  
- Operations are in R loops, with heavy string concatenation (`paste`) and `lapply`, which are inefficient for large-scale data.  

**Optimization Strategy**  
- Avoid per-row neighbor key computation; precompute neighbor relationships in an integer matrix or compressed form and reuse.  
- Use `data.table` for fast keyed joins and aggregations.  
- Reshape data to wide format by year, compute neighbor aggregations via vectorized operations or matrix indexing.  
- Compute all neighbor stats in a single pass rather than looping over each variable separately.  
- Optional: process in yearly chunks to fit memory.  

**Optimized Approach**  
- Use `data.table` with an edge list (`cell_id â†’ neighbor_id`).  
- Join neighbor values per year, then aggregate (max, min, mean) by source cell and year.  
- Merge results back to the main table.  

**Working R Code**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert rook_neighbors_unique (spdep::nb) into an edge list
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand edges across years to match panel
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, .(id = from, neighbor_id = to), by = .EACHI][rep(seq_len(.N), each = length(years))]
edges_expanded[, year := rep(years, times = nrow(edges))]

# Merge neighbor values
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

dt_neighbors <- merge(
  edges_expanded,
  cell_data[, c("id", "year", neighbor_vars), with = FALSE],
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  allow.cartesian = TRUE
)

# Compute stats per id-year for all variables
agg_list <- lapply(neighbor_vars, function(v) {
  list(max = as.name(v), min = as.name(v), mean = as.name(v))
})

agg_expr <- unlist(lapply(neighbor_vars, function(v) {
  list(
    (function(x) max(x, na.rm = TRUE))[[1]],
    (function(x) min(x, na.rm = TRUE))[[1]],
    (function(x) mean(x, na.rm = TRUE))[[1]]
  )
}), recursive = FALSE)

names(agg_expr) <- unlist(lapply(neighbor_vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))

neighbor_stats <- dt_neighbors[, agg_expr, by = .(id, year)]

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# Set NAs for rows without neighbors
```

**Key Advantages**  
- No huge lists in memory; uses efficient joins and aggregation.  
- Single pass computes all neighbor statistics, reducing runtime drastically.  
- Scales better and can be chunked by `year` if memory is tight (process each year separately and rbind).  

This approach should reduce runtime from ~86 hours to a few hours or less on a 16 GB machine.
END_RESPONSE>>>
