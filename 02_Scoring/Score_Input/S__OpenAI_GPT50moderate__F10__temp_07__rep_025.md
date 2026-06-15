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
run_id: OpenAI_F10_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is inefficient because:  
- `build_neighbor_lookup` is called once for all rows, but the neighbor lookup is recomputed for each observation without leveraging vectorization.  
- `compute_neighbor_stats` operates row-by-row and uses `lapply`, leading to ~6.46M iterations for every neighbor variable.  
- Repeated concatenation and lookup of keys (`paste`) and multiple traversals of large lists increase overhead.  
- Memory usage is high due to repeatedly expanding lists and copying large vectors.  
- The neighbor graph is static across years but not cached in an optimal way for fast querying.  

**Optimization Strategy**  
1. **Precompute graph topology once**: Store neighbor indices by cell ID only, then broadcast across years efficiently.  
2. **Vectorize aggregation**: Use `data.table` for fast joins and group operations rather than nested `lapply`.  
3. **Minimize string operations**: Replace key-based lookups with integer indexing.  
4. **Compute all neighbor stats in one pass**: For each variable, compute max/min/mean using data.table joins and aggregation instead of per-row iteration.  
5. **Memory efficiency**: Avoid copying large vectors repeatedly.  
6. Preserve numerical equivalence by computing the same aggregates (max, min, mean) for the same neighbor sets.  

---

### **Optimized Implementation in R**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (neighbors), indexed by id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# 1. Build static neighbor lookup (cell-to-cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i], nbr_id = id_order[rook_neighbors_unique[[i]]])
}))

setkey(neighbor_dt, src_id)

# 2. Expand by years using a cartesian join
years <- unique(cell_data$year)
neighbor_year_dt <- neighbor_dt[, .(year = years), by = .(src_id, nbr_id)]

# 3. Merge with cell_data twice: once for src_id-year, once for nbr_id-year
setkey(cell_data, id, year)

# Join to get neighbor values
neighbor_year_dt <- neighbor_year_dt[
  cell_data, on = .(nbr_id = id, year), nomatch = 0
]

# Now neighbor_year_dt has: src_id, nbr_id, year, and neighbor vars
# 4. Compute stats for each variable in one go
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_dt <- neighbor_year_dt[, lapply(.SD, function(x)
  list(max = if (length(x)) max(x, na.rm = TRUE) else NA_real_,
       min = if (length(x)) min(x, na.rm = TRUE) else NA_real_,
       mean = if (length(x)) mean(x, na.rm = TRUE) else NA_real_)
), by = .(src_id, year), .SDcols = neighbor_source_vars]

# Reshape aggregated stats
agg_dt <- rbindlist(lapply(neighbor_source_vars, function(var) {
  data.table(src_id = agg_dt$src_id, year = agg_dt$year,
             paste0(var, "_nbr_max") = sapply(agg_dt[[var]], `[[`, "max"),
             paste0(var, "_nbr_min") = sapply(agg_dt[[var]], `[[`, "min"),
             paste0(var, "_nbr_mean") = sapply(agg_dt[[var]], `[[`, "mean"))
}), use.names = TRUE, fill = TRUE)

# Combine by merging back to cell_data
setkey(agg_dt, src_id, year)
setkey(cell_data, id, year)
cell_data <- cell_data[agg_dt, on = .(id = src_id, year)]

# 5. Apply pre-trained Random Forest model
# Assuming rf_model is already loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Efficient**
- Builds neighbor graph once and reuses it.
- Uses `data.table` joins and aggregations instead of millions of `lapply` calls.
- Eliminates repeated string concatenation and key lookups.
- Computes all stats in a single grouped aggregation, massively reducing overhead.
- Preserves numerical equivalence (same max, min, mean for same neighbor sets).
- Scales well on 16GB RAM; expected runtime reduced from 86+ hours to a few hours or less.

**Expected Runtime**  
On ~6.46M rows and ~1.37M edges, this approach should complete in **under 2â€“4 hours** on a standard laptop depending on I/O speed and memory bandwidth.
END_RESPONSE>>>
