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
run_id: OpenAI_F09_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow because it repeatedly reconstructs neighbor relationships for each year and variable. With 6.46M rows, 5 variables, and 28 years, recomputing neighbor indices causes huge overhead. The neighbor structure is static across years, so rebuilding or iterating inefficiently on a per-variable basis inflates runtime. Additionally, the repeated `lapply` and indexing in R for millions of rows is costly.

---

**Optimization Strategy**  
1. **Build reusable neighbor lookup once**: Use cell IDs only (not year) so the structure is computed once and reused.
2. **Process one year at a time**: Join yearly attributes to the static neighbor structure to compute stats.
3. **Vectorize neighbor aggregation**: Use `data.table` or matrix operations for speed instead of millions of small `lapply` calls.
4. **Avoid repeated joins**: Precompute neighbor pairs as an edge table and join yearly values for aggregation.
5. **Preserve Random Forest model and estimand**: Only improve feature engineering performance, not model training.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor IDs for each cell in same order as id_order
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Step 1: Build static neighbor pairs table (once)
neighbor_pairs <- data.table(
  id      = rep(id_order, lengths(rook_neighbors_unique)),
  nbr_id  = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)
setkey(neighbor_pairs, nbr_id)

# Step 2: Convert cell_data to data.table and key by id, year
setDT(cell_data)
setkey(cell_data, id, year)

# Step 3: Compute neighbor stats by year and variable
compute_neighbor_features_year <- function(dt_year, neighbor_pairs, vars) {
  # Join yearly attributes to neighbor pairs
  pairs_year <- neighbor_pairs[dt_year, on = .(nbr_id = id)]
  # pairs_year now has: id (focal cell), nbr_id, year, vars...
  
  # Compute stats for each variable
  result_list <- lapply(vars, function(v) {
    agg <- pairs_year[, .(
      max = max(get(v), na.rm = TRUE),
      min = min(get(v), na.rm = TRUE),
      mean = mean(get(v), na.rm = TRUE)
    ), by = id]
    setnames(agg, c("id", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
    agg
  })
  
  # Merge all variable summaries for this year
  Reduce(function(x, y) merge(x, y, by = "id", all = TRUE), result_list)
}

# Step 4: Loop over years and merge back
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

neighbor_features_list <- lapply(years, function(y) {
  dt_year <- cell_data[year == y]
  feats <- compute_neighbor_features_year(dt_year, neighbor_pairs, vars)
  feats[, year := y]
  feats
})

neighbor_features <- rbindlist(neighbor_features_list)
setkey(neighbor_features, id, year)

# Step 5: Merge back into original cell_data
cell_data <- neighbor_features[cell_data]

# The cell_data now has neighbor-based features efficiently computed
```

---

**Why this is faster**  
- Neighbor relationships are computed once (O(n_neighbors)).
- Each year processes 344k rows instead of 6.46M in one go.
- Aggregation uses `data.table` group operations (highly optimized in C).
- Eliminates millions of small R loops.

This approach reduces runtime from **86+ hours to a few hours or less**, depending on I/O and CPU.
END_RESPONSE>>>
