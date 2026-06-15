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
run_id: OpenAI_F02_T10_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck arises from:  
1. Iterating over 6.46M rows with `lapply` in `build_neighbor_lookup` and `compute_neighbor_stats`.  
2. Repeated string concatenation (`paste`) and `setNames` combined with lookup overhead.  
3. Multiple passes over large vectors for 5 variables, triggering memory thrashing on a laptop.  

The structure creates *row-level neighbor expansions* for every observation, which scales poorly in pure R loops.  

**Optimization Strategy**  
- **Avoid repeated row-wise operations**: Precompute neighbor indices in a *vectorized or tabular form* to eliminate millions of `paste`.  
- **Use `data.table` for fast grouping and aggregation** instead of `lapply`.  
- **Reshape once** to `cell_id Ã— year`, then map neighbors using indexes rather than string keys.  
- **Chunk processing** to stay within 16â€¯GB RAM.  
- Use `parallel::mclapply` or `future.apply` if multi-core allowed.  
- Drop `do.call(rbind,...)` and build matrix directly for speed.  

**Working Optimized R Code**  

```r
library(data.table)

# Convert data to data.table for speed
setDT(cell_data)

# Ensure id_order and rook_neighbors_unique are available
# Precompute neighbor pairs as a long table
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  # i: reference cell, neigh: neighbor cell
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(neighbor_id = neighbor_id), by = .(id)]
neighbor_dt <- neighbor_dt[rep(seq_len(.N), each = length(years))]
neighbor_dt[, year := rep(years, times = .N / length(years))]

# Map cell-year rows to neighbor rows
cell_data_keyed <- cell_data[, .(id, year, row_id = .I)]
setkey(cell_data_keyed, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt[, neighbor_row := cell_data_keyed[J(neighbor_id, year), row_id]]

# Drop NA neighbor rows
neighbor_dt <- neighbor_dt[!is.na(neighbor_row)]

# Compute stats for multiple variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  
  # Lookup values for neighbor rows
  neighbor_dt[, val := vals[neighbor_row]]
  
  # Aggregate stats by (id, year)
  stats_dt <- neighbor_dt[!is.na(val),
                           .(max_val = max(val),
                             min_val = min(val),
                             mean_val = mean(val)),
                           by = .(id, year)]
  
  # Join back to cell_data
  setkey(stats_dt, id, year)
  cell_data[stats_dt, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats_dt, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats_dt, paste0(var_name, "_nbr_mean") := i.mean_val]
}
```

**Why Faster**  
- Eliminates 6.4M Ã— `lapply` loops; works entirely in grouped tabular operations.  
- Avoids repeated `paste`, `setNames`, and `do.call(rbind,...)`.  
- Uses efficient joins + aggregation in `data.table`, reducing runtime from hours to potentially under 1â€“2 hours on 16â€¯GB RAM.  
- Preserves model and numerical estimand because computations (max/min/mean) are unchanged.  

**Additional Tips**  
- Release intermediate objects with `rm()` and `gc()`.  
- If memory is still tight, split `neighbor_dt` or years into chunks and aggregate per chunk.  
- Optionally use disk-backed tables (e.g., `arrow` or `fst`) if needed for larger-than-memory processing.
END_RESPONSE>>>
