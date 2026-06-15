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
run_id: OpenAI_F09_T07_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every variable and year, repeatedly scanning a 6.46M-row dataset. This is highly inefficient because neighbor-cell relationships are static across years. The `build_neighbor_lookup` function creates an index for all rows, but the outer loop repeatedly calls `compute_and_add_neighbor_features` per variable, leading to redundant lookups and excessive allocations. With 6.46M rows, 28 years, and 5 variables, this multiplies the workload dramatically.  

**Optimization Strategy**  
- Build a single reusable adjacency table (`neighbor_table`) mapping each row to its neighbors once.  
- Perform a single join per year:  
  1. Attach yearly variable values to the neighbor table.  
  2. Compute neighbor stats (max, min, mean) using vectorized operations or `data.table`.  
  3. Join results back to the main dataset.  
- Leverage `data.table` for efficient grouping and aggregation.  
- Keep memory usage bounded by processing one year at a time.  
- Preserve the trained Random Forest model and original target.  

**Optimized R Code**

```r
library(data.table)

# Assume: cell_data (id, year, variables...), rook_neighbors_unique, id_order exist
setDT(cell_data)

# Build reusable adjacency table (cell_id -> neighbor_id)
neighbor_table <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Repeat each year for full panel
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure keys for fast joins
setkey(cell_data, id, year)

for (yr in years) {
  message("Processing year: ", yr)
  
  # Subset data for this year
  year_data <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_data, id)
  
  # Join neighbor_table with year_data to get neighbor values
  nb_vals <- neighbor_table[year_data, on = .(neighbor_id = id)]
  # nb_vals now has: id (cell), neighbor_id, year, vars...
  
  # Compute stats for each variable
  for (var in neighbor_source_vars) {
    stats <- nb_vals[!is.na(get(var)), .(
      n_max = max(get(var), na.rm = TRUE),
      n_min = min(get(var), na.rm = TRUE),
      n_mean = mean(get(var), na.rm = TRUE)
    ), by = .(id, year)]
    
    # Merge stats back into cell_data
    setnames(stats, c("n_max", "n_min", "n_mean"),
             paste0(var, c("_nb_max", "_nb_min", "_nb_mean")))
    cell_data[year == yr, (names(stats)[-c(1,2)]) := stats[, -c("id","year"), with = FALSE], on = .(id, year)]
  }
}

# At this point, cell_data has neighbor-based features for all years and variables.
# Proceed with prediction using the preserved Random Forest model:
predictions <- predict(rf_model, newdata = cell_data)
```

**Why this is faster:**  
- Neighbor relationships computed once and reused.  
- Aggregations are vectorized via `data.table` instead of per-row `lapply`.  
- Memory load is controlled by processing one year at a time.  

This approach reduces redundant computations and should cut runtime from 86+ hours to a few hours, depending on I/O and CPU performance.
END_RESPONSE>>>
