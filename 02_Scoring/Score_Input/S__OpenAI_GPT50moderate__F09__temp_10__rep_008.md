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
run_id: OpenAI_F09_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is extremely slow because `build_neighbor_lookup()` is repeatedly creating neighbor references per row and `compute_neighbor_stats()` loops over each of the 6.46M rows in pure R, each time performing lookups and aggregations. This results in massive overhead from millions of `lapply` calls and `paste()` operations. Core inefficiencies:  
- String concatenation (`paste`) for every row-year.  
- Redundant neighbor computation since the neighbor topology does not change over years.  
- Nested loops causing repeated work for each variable (5 times over all rows).  

**Optimization Strategy:**  
- Precompute a reusable adjacency table (`data.table`) of all cell-year â†’ neighbor-year pairs by performing a single `join` instead of millions of string ops.  
- Use `data.table` aggregation (`max`, `min`, `mean`) grouped by focal cell-year and variable.  
- Store the adjacency table once and reuse for all neighbor stats variables.  
- Avoid recomputing neighbor indices in R loops; perform vectorized joins.  

**Working R Code:**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build reusable adjacency table
# id_order: vector of all unique cell IDs consistent with rook_neighbors_unique
# rook_neighbors_unique: adjacency structure (list)
adj_list <- lapply(seq_along(id_order), function(i) {
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
})
adj_dt <- rbindlist(adj_list)
setkey(adj_dt, id)

# Expand to cell-year pairs
years <- sort(unique(cell_data$year))
adj_year <- adj_dt[CJ(year = years), on = .(id), allow.cartesian = TRUE]
setnames(adj_year, "year", "year")  # now columns: id, neighbor_id, year

# Merge neighbor attributes for all years
# Create a lookup table for merging
cell_key <- cell_data[, .(id, year, join_key = .I)]  # row index
setkey(cell_key, id, year)

adj_year <- adj_year[cell_key, on = .(neighbor_id = id, year)]
# adj_year now: id (focal), neighbor_id, year, join_key (row index in cell_data)

# For all neighbor source vars, compute stats in a single pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_source_vars) {
  vals <- cell_data[[v]]
  adj_year[, neighbor_val := vals[join_key]]
  
  # Aggregate by focal cell-year
  stats <- adj_year[!is.na(neighbor_val),
    .(
      max_val = max(neighbor_val, na.rm = TRUE),
      min_val = min(neighbor_val, na.rm = TRUE),
      mean_val = mean(neighbor_val, na.rm = TRUE)
    ),
    by = .(id, year)
  ]
  
  # Merge back into cell_data
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  
  cell_data <- stats[cell_data]
  setnames(cell_data,
    c("max_val", "min_val", "mean_val"),
    paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))
  )
}

# Final dataset is cell_data with new neighbor stats columns
# Random Forest prediction unchanged: use predict(rf_model, newdata = cell_data)
```

**Expected performance improvement:**  
- String operations eliminated, replaced with keyed joins.
- Fully vectorized aggregation using `data.table` reduces complexity drastically.
- Adjacency expansion is done **once** (not per variable).
- On a 16 GB machine, this approach should bring 86 hours down to a few hours (or less) if chunked properly.  

**Key principles preserved:**  
- Trained Random Forest remains unchanged.
- Same estimand (identical neighbor max/min/mean).  
- Memory controlled by processing in chunks if needed.
END_RESPONSE>>>
