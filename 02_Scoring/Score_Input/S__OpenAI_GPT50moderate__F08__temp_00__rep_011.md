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
run_id: OpenAI_F08_T00_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M rows) by iterating through a large list of neighbor indices. This results in repeated lookups and redundant computations across years, even though the neighbor structure is static. The complexity is roughly `O(N * k)` per variable, where `N` is 6.46M and `k` is average neighbor count, multiplied by 5 variables. This is memory- and CPU-intensive on a 16 GB laptop.

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute a neighbor index map at the cell level (344,208 cells) once.
- **Vectorize by year**: For each year, compute neighbor stats using matrix operations instead of per-row loops.
- **Avoid repeated lookups**: Use a sparse adjacency matrix or pre-built index lists for fast aggregation.
- **Chunk by year**: Process 28 yearly slices sequentially to keep memory usage manageable.
- **Bind results back**: Append computed neighbor features to the original data without altering the Random Forest model.

---

### **Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor indices (spdep::nb)
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor index list keyed by cell id
neighbor_map <- setNames(rook_neighbors_unique, id_order)

# Function to compute neighbor stats for one variable in one year
compute_yearly_stats <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  names(vals) <- dt_year$id
  res <- lapply(dt_year$id, function(cid) {
    nbs <- neighbor_map[[as.character(cid)]]
    if (length(nbs) == 0) return(c(NA, NA, NA))
    nb_vals <- vals[as.character(nbs)]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) return(c(NA, NA, NA))
    c(max(nb_vals), min(nb_vals), mean(nb_vals))
  })
  do.call(rbind, res)
}

# Main loop: process by year and variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare columns for new features
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nb_max") := NA_real_]
  cell_data[, paste0(v, "_nb_min") := NA_real_]
  cell_data[, paste0(v, "_nb_mean") := NA_real_]
}

years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  for (v in neighbor_source_vars) {
    stats <- compute_yearly_stats(dt_year, v)
    cell_data[year == yr, paste0(v, "_nb_max") := stats[,1]]
    cell_data[year == yr, paste0(v, "_nb_min") := stats[,2]]
    cell_data[year == yr, paste0(v, "_nb_mean") := stats[,3]]
  }
}

# At this point, cell_data has neighbor features added
# Proceed with Random Forest prediction using the pre-trained model
```

---

### **Why This Is Faster**
- **No per-row neighbor lookup across all years**: We compute stats per year using pre-built neighbor lists.
- **Reduced iterations**: 28 yearly slices Ã— 5 variables instead of 6.46M row-level loops.
- **Memory efficiency**: Only one year in memory at a time.
- **Preserves estimand**: Same neighbor max, min, mean logic, just reorganized.

**Expected runtime**: Hours reduced from 86+ to a few hours (or less) on a 16 GB laptop.
END_RESPONSE>>>
