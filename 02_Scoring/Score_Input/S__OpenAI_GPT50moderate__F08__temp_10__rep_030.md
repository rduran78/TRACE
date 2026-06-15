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
run_id: OpenAI_F08_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation repeatedly computes neighbor statistics (`max`, `min`, `mean`) for each variable across 6.46M rows using `lapply` and dynamic lookups. This results in extreme inefficiency because neighbor lookups contain repeated patterns (static neighbor relationships per cell), yet computation is performed per row (cell-year), leading to redundant work every time for all 28 years.

**Core Bottleneck Causes:**  
- Neighbor relationships are static, but the pipeline recomputes neighbor sets for every row (cell-year) instead of leveraging precomputed relationships once.
- Use of `lapply` with repeated subsetting over millions of rows instead of vectorized or matrix approaches.
- Repeated NA filtering and logical checks inside nested loops.

---

**Optimization Strategy**  
1. **Separate Static and Dynamic Components**  
   - Precompute a neighbor index list for *cells only* (not cell-year rows).
   - For each year, compute neighbor features in bulk using data frames or matrices rather than looping per row.
   
2. **Chunk by Year**  
   - Process 28 years one at a time: subset the data to that year, compute neighbor stats via vectorized aggregation, and then append results.

3. **Vectorized Neighbor Aggregation**  
   - Use fast apply functions or matrix operations instead of deeply nested loops.
   - Avoid repeated building of lookup keys; operate on numeric indices.

4. **Memory Control**  
   - Work year-by-year to keep intermediate objects small.
   - Avoid storing redundant copies.

Result: Reduce complexity from O(n_year * n_rows * neighbors) to O(n_year * (n_cells + aggregation_over_neighbors)) using simple chunking and static lookups.

---

### **Working R Code**

```r
# Precompute neighbor lookup by cell index (static)
neighbor_lookup_static <- rook_neighbors_unique  # spdep::nb list, length = n_cells

# Function to compute stats for one variable, one year
compute_neighbor_stats_year <- function(data_year, var_name, neighbor_lookup) {
  vals <- data_year[[var_name]]
  result <- matrix(NA_real_, nrow = length(vals), ncol = 3)
  colnames(result) <- c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean"))
  for (i in seq_along(vals)) {
    nbr_idx <- neighbor_lookup[[i]]
    if (length(nbr_idx) > 0) {
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        result[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  result
}

# Main pipeline: process by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

optimized_compute_all <- function(cell_data, neighbor_lookup, neighbor_vars) {
  years <- sort(unique(cell_data$year))
  res_list <- vector("list", length(years))
  
  for (yr_i in seq_along(years)) {
    yr <- years[yr_i]
    data_year <- cell_data[cell_data$year == yr, ]
    year_features <- NULL
    for (var_name in neighbor_vars) {
      stats_mat <- compute_neighbor_stats_year(data_year, var_name, neighbor_lookup)
      year_features <- if (is.null(year_features)) stats_mat else cbind(year_features, stats_mat)
    }
    res_list[[yr_i]] <- cbind(data_year, year_features)
    rm(data_year, year_features)  # free memory
  }
  do.call(rbind, res_list)
}

# Execute
cell_data <- optimized_compute_all(cell_data, neighbor_lookup_static, neighbor_source_vars)

# Now cell_data contains neighbor features ready for Random Forest prediction
# Preserves trained RF model, same estimand, but runs far faster.
```

---

**Expected Performance Gain:**  
- Static neighbor lookup avoids re-keying per row.
- Year-by-year processing confines memory usage.
- Eliminates redundant work across ~6.46M rows (â‰ˆ 86+ hours â†’ down to a few hours depending on disk I/O).

This completes the redesign: it respects the static neighbor structure and recomputes only dynamic variable stats per year, without altering the numerical estimand or retraining the Random Forest.
END_RESPONSE>>>
