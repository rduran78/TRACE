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
run_id: OpenAI_F10_T03_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor lookups and stats for each cell-year row using `lapply` over millions of rows.  
- For each variable, it loops through all rows and performs indexing, causing redundant work and poor memory locality.  
- Graph topology is recomputed for every year instead of being reused.  
- No vectorization; heavy reliance on R-level loops.  

**Optimization Strategy**  
- Precompute and store graph topology (neighbor indices) once for all cells.  
- Use matrix operations and `data.table` for fast grouping and joins.  
- Compute neighbor statistics year by year using efficient aggregation, avoiding per-row `lapply`.  
- Avoid repeated paste operations; use integer keys for joins.  
- Process variables in bulk rather than one at a time when possible.  
- Preserve numerical equivalence by using the same max, min, mean definitions.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor graph topology (once)
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
id_order <- sort(unique(cell_data$id))
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build a long neighbor mapping table: (id, neighbor_id)
neighbor_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand across years: join with years from cell_data
years <- sort(unique(cell_data$year))
neighbor_pairs_expanded <- neighbor_pairs[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Merge neighbor attributes
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor attributes for each year
neighbor_data <- merge(
  neighbor_pairs_expanded,
  cell_data[, c("id", "year", vars), with = FALSE],
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  all.x = TRUE
)

# Compute stats per id-year
agg_stats <- neighbor_data[, lapply(.SD, function(x) {
  if (all(is.na(x))) return(c(NA_real_, NA_real_, NA_real_))
  c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))
}), by = .(id, year), .SDcols = vars]

# Reshape: create columns for each var and stat
stat_names <- c("max", "min", "mean")
new_cols <- unlist(lapply(vars, function(v) paste(v, stat_names, sep = "_")))
agg_stats <- agg_stats[, as.list(unlist(.SD)), by = .(id, year), .SDcols = vars]
setnames(agg_stats, c("id", "year", new_cols))

# Merge back to cell_data
cell_data <- merge(cell_data, agg_stats, by = c("id", "year"), all.x = TRUE)

# Apply pre-trained Random Forest model (preserve original estimand)
# Assume rf_model is already loaded
predictions <- predict(rf_model, newdata = cell_data)
```

**Key Improvements**  
- Graph topology built once (`neighbor_pairs`).  
- Vectorized expansion across years using `data.table`.  
- Single aggregation step for all variables.  
- Eliminated per-row loops and repeated string operations.  

This approach reduces complexity from millions of `lapply` calls to efficient joins and aggregations, making it feasible on a 16 GB laptop.
END_RESPONSE>>>
