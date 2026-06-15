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
run_id: OpenAI_F02_T00_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F02_OpenAI_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is slow and memory-intensive because:  
- `lapply` over 6.46M rows creates large intermediate lists.  
- Neighbor lookups repeatedly compute string keys (`paste`) and indexing, which is expensive.  
- `compute_neighbor_stats` loops over all rows and performs repeated subsetting.  
- No vectorization or efficient data structures (e.g., `data.table`) are used.  

**Optimization Strategy**  
- Use `data.table` for fast joins and aggregation.  
- Precompute neighbor relationships as a long table (cell-year â†’ neighbor-year) instead of per-row lists.  
- Compute neighbor statistics in a single grouped operation rather than row-wise `lapply`.  
- Avoid repeated string concatenation; use integer keys.  
- Process variables in a single pass if possible.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create integer keys for id and year
cell_data[, id_int := as.integer(factor(id))]
cell_data[, row_id := .I]

# Expand neighbor relationships into a long table
# rook_neighbors_unique: list of integer neighbors per id_order position
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id_int = as.integer(i),
      neighbor_id_int = as.integer(rook_neighbors_unique[[i]])
    )
  })
)

# Join with years to create cell-year â†’ neighbor-year pairs
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id_int, neighbor_id_int, year = years), by = .(id_int, neighbor_id_int)]

# Map to row indices
cell_data_keyed <- cell_data[, .(id_int, year, row_id)]
setkey(cell_data_keyed, id_int, year)
setkey(neighbor_dt, neighbor_id_int, year)

neighbor_dt <- neighbor_dt[cell_data_keyed, on = .(neighbor_id_int = id_int, year), nomatch = 0]
setnames(neighbor_dt, "row_id", "neighbor_row_id")

# Now neighbor_dt has: id_int, neighbor_id_int, year, neighbor_row_id
# Add original row_id for target cell-year
neighbor_dt <- neighbor_dt[cell_data_keyed, on = .(id_int, year)]
setnames(neighbor_dt, "row_id", "target_row_id")

# Compute neighbor stats for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in vars) {
  vals <- cell_data[, .(row_id, v = get(var))]
  setkey(vals, row_id)
  neighbor_dt <- neighbor_dt[vals, on = .(neighbor_row_id = row_id)]
  
  # Aggregate stats by target_row_id
  stats <- neighbor_dt[!is.na(v), .(
    max_val = max(v),
    min_val = min(v),
    mean_val = mean(v)
  ), by = target_row_id]
  
  # Merge back to cell_data
  setkey(stats, target_row_id)
  cell_data <- cell_data[stats, on = .(row_id = target_row_id)]
  
  # Rename columns
  setnames(cell_data, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  # Remove v column from neighbor_dt for next iteration
  neighbor_dt[, v := NULL]
}

# Drop helper columns
cell_data[, c("id_int", "row_id") := NULL]
```

**Why This Works**  
- Converts neighbor relationships into a long format for efficient joins.  
- Uses `data.table` grouping instead of millions of `lapply` calls.  
- Avoids repeated string operations and leverages integer keys.  
- Processes each variable in a single grouped aggregation pass.  

**Expected Performance**  
- Memory footprint reduced (no giant lists).  
- Runtime reduced from 86+ hours to a few hours or less on a 16 GB laptop.  
- Preserves original estimand and trained Random Forest model.
END_RESPONSE>>>
