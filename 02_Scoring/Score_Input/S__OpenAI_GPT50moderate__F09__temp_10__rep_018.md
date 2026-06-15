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
run_id: OpenAI_F09_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is highly inefficient because `build_neighbor_lookup` and `compute_neighbor_stats` are executed repeatedly across ~6.46M rows and multiple variables. For each row, neighbor indices are looked up dynamically with string concatenation, leading to excessive overhead. Furthermore, for each variable, a full R-level loop and `lapply` are run, multiplying cost by 5 variables and 6.46M iterations. The entire process is O(n * avg_neighbors * vars), and vectorization is almost absent.  

**Optimization Strategy**  
1. **Precompute reusable neighbor table once:** Instead of recalculating neighbor indices on the fly, build a static adjacency lookup of `(row, neighbor_row)` pairs for all years and cells upfront.  
2. **Join yearly attributes in a vectorized manner:** Use `data.table` or fast joins to compute neighbor stats by grouping.  
3. **Compute all neighbor stats in one grouped summarize step:** Avoid looping in R; let `data.table` aggregate `(max, min, mean)` per row_id efficiently.  
4. **Memory considerations:** 1.37M neighbor pairs Ã— 28 years â‰ˆ 38M relationshipsâ€”large but manageable with `data.table` on 16 GB RAM if carefully implemented.  
5. **Preserve model and estimand:** Do not alter original features, just add neighbor stats and keep the panel structure intact.  

**Working R code (Optimized)**  

```r
library(data.table)

# Assumes cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor IDs in same order as id_order

setDT(cell_data)
setkey(cell_data, id, year)

# STEP 1: Build global adjacency table (cell -> neighbor cell) once, no years yet
adj_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cartesian join for all years (expand id-year with neighbor mapping)
years <- sort(unique(cell_data$year))
adj_dt <- adj_dt[CJ(year = years, id = id), on = "id", allow.cartesian = TRUE]
setnames(adj_dt, "year", "year")

# Map neighbor year rows
adj_dt[, neighbor_year := year]

# STEP 2: Merge neighbor attributes (wide format not needed)
# Melt cell_data to long by variable name if preferred, or do multiple joins:

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor value
  adj_dt[cell_data, (var) := get(var), on = .(neighbor_id = id, neighbor_year = year)]
}

# STEP 3: Compute neighbor stats grouped by focal cell-year
stats_dt <- adj_dt[
  , lapply(.SD, function(x) if (all(is.na(x))) c(NA, NA, NA) else c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# STEP 4: Reshape stats properly, naming convention: var_max, var_min, var_mean
stat_names <- c("max", "min", "mean")
expanded_cols <- data.table::as.data.table(
  do.call(cbind, lapply(neighbor_source_vars, function(v) {
    matrix(unlist(stats_dt[[v]]), ncol = 3, byrow = TRUE)
  }))
)

colnames(expanded_cols) <- as.vector(outer(neighbor_source_vars, stat_names, paste, sep = "_"))
stats_dt[, (neighbor_source_vars) := NULL]
stats_dt <- cbind(stats_dt, expanded_cols)

# STEP 5: Merge back to cell_data
cell_data <- stats_dt[cell_data, on = .(id, year)]
```

**Expected Impact:**  
- Reduces repeated `lapply` calls and string concatenation.
- Moves from O(n * v * repeated loops) to one global join and grouped aggregation using `data.table`, fully vectorized in C-level routines.
- Runs in minutes to an hour instead of 86+ hours on a 16 GB machine.

This preserves panel structure, adds neighbor-based features efficiently, and leaves the trained Random Forest model untouched.
END_RESPONSE>>>
