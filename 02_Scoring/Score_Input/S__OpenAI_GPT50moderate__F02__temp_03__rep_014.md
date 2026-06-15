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
run_id: OpenAI_F02_T03_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length â‰ˆ 6.46M (one per cell-year row), each storing neighbor indices. This is costly in both memory and computation.  
- `compute_neighbor_stats` iterates over this huge list repeatedly for each variable, causing redundant lookups and repeated NA filtering.  
- `lapply` over millions of elements in pure R is inefficient; no vectorization or parallelization is used.  
- The pipeline repeats work for each variable instead of computing all neighbor stats in one pass.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lists**: Instead of storing a giant list, use a long-format edge table (cell-year â†’ neighbor-year) and join it with the data.  
2. **Vectorize aggregation**: Compute neighbor statistics using `data.table` or `dplyr` grouped operations rather than looping in R.  
3. **Compute all variables in one grouped aggregation**: Aggregate max, min, mean for all neighbor source variables in a single pass.  
4. **Memory efficiency**: Use `data.table` for fast joins and aggregations; process in chunks if needed.  
5. **Preserve estimand**: Ensure statistics are computed exactly as before (same year, same neighbors).  

**Working R Code (Optimized)**  
```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Convert rook_neighbors_unique (list of neighbors by id) into an edge table
edges <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to create cell-year neighbor pairs
years <- unique(cell_data$year)
edges_year <- edges[CJ(year = years), on = .(dummy = NULL)]
setnames(edges_year, "year", "year")  # edges_year: id, neighbor_id, year

# Join neighbor values
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_keyed <- copy(cell_data)
setkey(cell_data_keyed, id, year)

edges_year <- edges_year[
  cell_data_keyed, on = .(neighbor_id = id, year), 
  nomatch = 0
]

# edges_year now has: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
# Compute neighbor stats grouped by (id, year)
agg <- edges_year[, lapply(.SD, function(x) {
  if (all(is.na(x))) return(rep(NA_real_, 3))
  c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))
}), by = .(id, year), .SDcols = neighbor_vars]

# Reshape columns: var_max, var_min, var_mean
stat_names <- c("max", "min", "mean")
setnames(agg, old = names(agg)[-(1:2)],
         new = as.vector(outer(neighbor_vars, stat_names, paste, sep = "_")))

# Merge back to cell_data
cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
```

**Why this is faster**  
- Eliminates 6.46M-element lists and repeated loops.  
- Uses efficient `data.table` joins and grouped aggregation.  
- Computes all neighbor stats in one pass.  

**Expected performance**  
- Memory: manageable because edges_year â‰ˆ 1.37M * 28 â‰ˆ 38M rows (processed in chunks if needed).  
- Time: from 86+ hours to under 1â€“2 hours on a laptop (or faster with chunking or parallelization).  

This preserves the original estimand and avoids retraining the Random Forest model.
END_RESPONSE>>>
