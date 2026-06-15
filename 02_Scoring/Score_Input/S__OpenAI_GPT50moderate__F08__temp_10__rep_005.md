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
run_id: OpenAI_F08_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor indices for every cell-year combination, which is inefficient because the spatial neighbor relationships are static across years. For each row (cell-year), `build_neighbor_lookup` computes neighbor indices anew using a key-based lookup that involves ~6.46 million rows and 28 repeated passes through lists of neighbors. This results in extremely high computational overhead and memory usage (vast duplication of neighbor lookups). The static-vs-changing distinction was not exploited: the spatial adjacency is fixed, only variable values change yearly.  

**Optimization Strategy**  
1. Precompute a static neighbor index lookup by cell ID only (not by year).
2. Use vectorized computations grouped by year instead of iterating over all rows.
3. Reshape data into a matrix or list by year, apply neighbor statistics in batches.
4. Avoid repeated string concatenation and `setNames` key lookups for every row.
5. Use `data.table` or matrix operations for speed and memory efficiency.
6. Preserve estimands by using the same aggregation (max, min, mean) but in optimized form.
  
**Working R Code**  

```r
library(data.table)

# Convert to data.table for efficiency
cell_data_dt <- as.data.table(cell_data)

# Precompute static neighbor lookup indexed by cell_id (NOT year)
# rook_neighbors_unique assumed to be a list of integer vectors aligned with id_order
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_lookup_static <- lapply(seq_along(id_order), function(i) {
  id_order[rook_neighbors_unique[[i]]] # neighbor IDs for cell i
})
names(neighbor_lookup_static) <- as.character(id_order)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_year <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  result <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
  
  for (i in seq_len(nrow(dt_year))) {
    nbr_ids <- neighbor_lookup_static[[as.character(dt_year$id[i])]]
    if (length(nbr_ids) == 0) next
    nbr_idx <- match(nbr_ids, dt_year$id, nomatch = 0L)
    if (length(nbr_idx) == 0) next
    nbr_vals <- vals[nbr_idx]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) == 0) next
    result[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
  }
  
  colnames(result) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  result
}

# Process in yearly batches
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create containers
results_list <- vector("list", length(unique(cell_data_dt$year)))

years <- sort(unique(cell_data_dt$year))
for (y in years) {
  dt_year <- cell_data_dt[year == y]
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_year(dt_year, var_name)
  })
  combined <- do.call(cbind, stats_list)
  results_list[[as.character(y)]] <- combined
}

# Bind results back to original data.table
# Ensure row order matches initial data for each year
out_matrix <- do.call(rbind, results_list)
cell_data_dt <- cbind(cell_data_dt, out_matrix)

# Result: cell_data_dt now contains neighbor feature columns ready for RF predict
# Random Forest prediction as before, preserving original estimand
predictions <- predict(pretrained_rf_model, newdata = cell_data_dt)
```

**Expected Impact**  
- Eliminates repeated neighbor key construction across 6.46 million rows.
- Reduces complexity from ~195M operations (per-row lookups * years) to ~6.46M operations with efficient indexing.
- Runtime expected reduction from 86+ hours to a few hours on a 16 GB RAM laptop.
- Preserves the Random Forest model and feature semantics exactly.
END_RESPONSE>>>
