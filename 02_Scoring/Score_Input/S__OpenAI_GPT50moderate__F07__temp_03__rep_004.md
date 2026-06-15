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
run_id: OpenAI_F07_T03_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses **nested `lapply` loops** over ~6.46M rows, repeatedly performing string concatenation, lookups, and list indexing.  
- `build_neighbor_lookup` constructs neighbor indices for every row by recomputing keys per year, which is very expensive.  
- `compute_neighbor_stats` iterates row-wise, causing huge overhead in Râ€™s interpreter.  
- Memory pressure: 6.46M rows Ã— 5 variables Ã— 3 stats = ~97M values.  

**Optimization Strategy**  
- **Precompute a long-format neighbor table**: Expand neighbors across years once, join with data using integer keys instead of strings.  
- Use **`data.table`** for fast joins and grouped aggregation (vectorized, compiled C backend).  
- Compute max, min, mean in a single grouped operation per variable.  
- Avoid loops over rows; instead, operate on the entire dataset in chunks or fully vectorized.  
- Keep the Random Forest model unchanged; only generate new features efficiently.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data (data.frame) with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (directed) for all cells
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Cross with years to create full panel neighbor mapping
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = src, nbr, year = years), by = .(src, nbr)]

# Join neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, nbr, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(nbr = id, year), nomatch = 0]

# neighbor_dt now has: src, nbr, year, and all variables from cell_data for nbr
# Compute stats per src-year
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(vars, function(v) list(
  max = as.name(v),
  min = as.name(v),
  mean = as.name(v)
))

# Build aggregation expression
agg_expr <- unlist(lapply(vars, function(v) {
  list(
    as.call(list(as.name("max"), as.name(v), na.rm = TRUE)),
    as.call(list(as.name("min"), as.name(v), na.rm = TRUE)),
    as.call(list(as.name("mean"), as.name(v), na.rm = TRUE))
  )
}))
agg_names <- unlist(lapply(vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))

stats_dt <- neighbor_dt[, eval(as.call(c(as.name("list"), agg_expr))), by = .(id = src, year)]
setnames(stats_dt, old = names(stats_dt)[-(1:2)], new = agg_names)

# Merge back to cell_data
cell_data <- stats_dt[cell_data, on = .(id, year)]

# cell_data now has neighbor max/min/mean for all 5 variables
```

---

**Why This Works**  
- **Single pass aggregation** using `data.table` instead of millions of row-wise loops.  
- Integer joins instead of string concatenation.  
- Memory-efficient because operations are vectorized and in-place.  

**Expected Performance**  
- From 86+ hours â†’ **minutes to a few hours** on 16 GB RAM.  
- Preserves original rook-neighbor relationships and numerical estimands.  
- No retraining of Random Forest model requiredâ€”just adds new features.
END_RESPONSE>>>
