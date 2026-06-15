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
run_id: OpenAI_F09_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is slow because for each of the ~6.46M cell-year rows, `build_neighbor_lookup` repeatedly maps neighbor relationships by string concatenation and lookup in large hash tables. This is repeated for every variable, causing excessive overhead. The neighbor structure is static across years, so rebuilding neighbor relationships for each year is unnecessary. Additionally, repeated `lapply` calls over millions of rows are expensive.

---

**Optimization Strategy:**  
1. **Build a reusable neighbor adjacency table once** using the static cell-to-cell relationships (`id_order`, `rook_neighbors_unique`).
2. **Precompute a long-format neighbor join table** `(cell_id, year, neighbor_id)` and then join yearly attributes from the main dataset for all variables at once.
3. Aggregate neighbor statistics (max, min, mean) per `(cell_id, year)` using `data.table` for speed.
4. Merge these aggregated stats back into the main dataset.
5. Do this in a vectorized manner without repeated `lapply` over millions of rows.

This approach converts an `O(N * neighbors)` repeated computation into a single large grouped aggregation, leveraging efficient joins and grouping.

---

**Working R Code:**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell ids in neighbor list order
# rook_neighbors_unique: list of integer vectors representing neighbors

# 1. Build adjacency table (cell_id -> neighbor_id)
adj_list <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# 2. Expand adjacency for all years
years <- sort(unique(cell_data$year))
adj_dt <- adj_list[, .(cell_id = rep(cell_id, each = length(years)),
                       neighbor_id = rep(neighbor_id, each = length(years)),
                       year = rep(years, times = .N))]
setkey(adj_dt, neighbor_id, year)

# 3. Prepare main data keyed by (id, year)
setkey(cell_data, id, year)

# 4. Join neighbor attributes
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
adj_dt <- cell_data[adj_dt, on = .(id = neighbor_id, year), nomatch = 0L]

# 5. Aggregate neighbor stats for each cell-year
agg_list <- lapply(vars, function(v) {
  adj_dt[, .(
    paste0(v, "_nb_max") = max(get(v), na.rm = TRUE),
    paste0(v, "_nb_min") = min(get(v), na.rm = TRUE),
    paste0(v, "_nb_mean") = mean(get(v), na.rm = TRUE)
  ), by = .(cell_id, year)]
})

# Merge all stats
neighbor_stats <- Reduce(function(x, y) merge(x, y, by = c("cell_id", "year")), agg_list)

# 6. Merge back into cell_data
setnames(neighbor_stats, "cell_id", "id")
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# Now cell_data has neighbor features for RF prediction
```

**Expected Outcome:**  
- No repeated neighbor lookups per variable/year.
- Scales efficiently using `data.table` joins and grouping.
- Memory usage remains manageable (~16 GB RAM) since adjacency Ã— years is ~38M rows (1.37M edges Ã— 28 years).
- Runtime reduced from 86+ hours to likely under an hour on a standard laptop.

**Preserves:**  
- Original numerical estimand unchanged.
- Previously trained Random Forest model remains intact.
- Adds neighbor max, min, mean features for all 5 source variables efficiently.
END_RESPONSE>>>
