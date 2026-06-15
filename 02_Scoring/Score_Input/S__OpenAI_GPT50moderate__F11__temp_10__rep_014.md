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
run_id: OpenAI_F11_T10_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect. The pipeline bottleneck is not Random Forest inference (which is typically very fast on an already-trained model). Instead, the audit shows extreme time complexity in building and applying neighbor lookups and computing neighbor statisticsâ€”both use repeated **`lapply` over millions of rows** with heavy string manipulation and repeated set lookups, creating huge overhead.

### **Diagnosis**
- `build_neighbor_lookup` constructs neighbor index vectors for each of ~6.46M rows using repeated `paste()` and named vector lookups.
- `compute_neighbor_stats` similarly loops through all neighbors for each row, doing repeated NA filtering and aggregations.
- These steps scale with cell-year rows Ã— neighbors and dominate the runtime. This is a classic case of Râ€™s `lapply` overhead and string-based keys causing massive inefficiency.

### **Optimization Strategy**
Replace expensive row-wise `lapply` and string lookups with:
1. **Precompute a fast integer mapping** (avoid `paste()`, work with `id` and `year` as integers).
2. Use **`data.table` join and aggregation** or **vectorized computations** instead of nested loops.
3. Compute neighbor statistics in one grouped operation instead of per-row iteration.

### **Optimized Implementation**

```r
library(data.table)

# Convert data to data.table
setDT(cell_data)

# Ensure an integer key for quick join
cell_data[, cell_year_id := .I]           # unique row id
cell_data[, id_year_key := paste(id, year, sep = "_")]

# Build data.table of all (cell, neighbor) relationships across years
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  data.table(
    source_id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}), use.names = TRUE)

# Expand for all years
years <- unique(cell_data$year)
expanded_pairs <- neighbor_pairs[, .(source_id, neighbor_id, year = years), by = .(source_id, neighbor_id)]

# Map to cell_year_id for source and neighbor
expanded_pairs[, source_key := paste(source_id, year, sep = "_")]
expanded_pairs[, neighbor_key := paste(neighbor_id, year, sep = "_")]

expanded_pairs[, source_row := cell_data[expanded_pairs$source_key, on = .(id_year_key), which = TRUE]]
expanded_pairs[, neighbor_row := cell_data[expanded_pairs$neighbor_key, on = .(id_year_key), which = TRUE]]

# Drop NAs
expanded_pairs <- expanded_pairs[!is.na(source_row) & !is.na(neighbor_row)]

# Function to compute neighbor stats for each variable in one pass
compute_neighbor_stats_dt <- function(var) {
  vals <- cell_data[[var]]
  stats <- expanded_pairs[, .(
    max_val = max(vals[neighbor_row], na.rm = TRUE),
    min_val = min(vals[neighbor_row], na.rm = TRUE),
    mean_val = mean(vals[neighbor_row], na.rm = TRUE)
  ), by = source_row]
  stats
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge neighbor features
for (var in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(var)
  cell_data[stats$source_row, paste0(var, "_nb_max") := stats$max_val]
  cell_data[stats$source_row, paste0(var, "_nb_min") := stats$min_val]
  cell_data[stats$source_row, paste0(var, "_nb_mean") := stats$mean_val]
}
```

### **Why This Works**
- Eliminates string concatenation inside massive loops.
- Reduces time complexity from millions of row-wise operations to efficient joins and grouped aggregations.
- Uses `data.table`, optimized for large datasets in memory.

### **Expected Speedup**
- From 86+ hours down to minutes or a few hours on a standard laptop.
- Memory footprint manageable within 16 GB since operations are vectorized.

**Preserves trained Random Forest model and original estimand, but removes the real bottleneck: neighbor feature computation.**  
Final inference step with Random Forest will be trivial afterwards.
END_RESPONSE>>>
