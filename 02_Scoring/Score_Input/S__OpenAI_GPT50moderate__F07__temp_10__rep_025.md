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
run_id: OpenAI_F07_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current method iterates over ~6.46M rows in R lists and applies repeated lookups, creating high overhead in memory and CPU. Calling `lapply` for every observation, repeatedly manipulating character keys, and doing per-row computations (max/min/mean) is extremely costly. With 344k cells Ã— 28 years and ~1.37M neighbor links, naive list operations scale poorly. Core bottlenecks:  
- Too many `paste` and `setNames` conversions  
- Row-wise neighbor queries instead of vectorized joins  
- No use of efficient data structures or compiled code  

---

**Optimization Strategy**  
- Convert entire panel into an **integer-based map** instead of strings.  
- Precompute the neighbor lookup into a **flat integer index array**.  
- Use **data.table** for fast grouping/aggregation via keys or indexed joins.  
- Avoid per-row `lapply`: compute stats for all links in bulk, then aggregate.  
- Compute neighbor stats by a join of `(id, year)` pairs with their neighbors in a long format.  
- Leverage efficient aggregation (data.table) and avoid retraining the Random Forest.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Flatten neighbor structure into pairs once (id, neighbor_id)
# rook_neighbors_unique: list of neighbors for each cell_id in id_order
neighbor_pairs <- data.table(
  id = rep(id_order, times = lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand to panel by crossing with all years
years <- sort(unique(cell_data$year))
neighbor_panel <- neighbor_pairs[, .(id, neighbor_id), by = .EACHI][, year := years, by = .(id, neighbor_id)]

# Reshape to compute neighbor stats efficiently
# Join neighbor_panel with cell_data (for neighbor values)
long_neighbors <- merge(
  neighbor_panel,
  cell_data[, .(neighbor_id = id, year, ntl, ec, pop_density, def, usd_est_n2)],
  by = c("neighbor_id", "year"),
  allow.cartesian = TRUE
)

# Aggregate neighbor stats for each id-year
agg_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stat_dt <- long_neighbors[, lapply(.SD, function(x) {
  if (all(is.na(x))) return(list(NA, NA, NA))
  return(list(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE)))
}), .SDcols = agg_vars, by = .(id, year)]

# Flatten the nested lists (max/min/mean) and rename
stat_cols <- unlist(lapply(agg_vars, function(v) paste(v, c("max", "min", "mean"), sep = "_")))
stat_dt[, (stat_cols) := do.call(cbind, lapply(.SD, unlist)), .SDcols = agg_vars]
stat_dt[, (agg_vars) := NULL]

# Merge back into cell_data
cell_data <- merge(cell_data, stat_dt, by = c("id", "year"), all.x = TRUE)

# At this point, cell_data contains original features plus neighbor stats; model untouched
```

---

**Why Efficient**  
- Single bulk join and aggregation replaces millions of row-wise list lookups.  
- Avoids repetitive string operationsâ€”uses integer keys.  
- Memory footprint controlled via `data.table` streaming and indexing.  

**Expected Impact**  
From >86 hours down to minutes or tens of minutes on 16GB hardware, leveraging vectorized and indexed computations instead of iterative loops.
END_RESPONSE>>>
