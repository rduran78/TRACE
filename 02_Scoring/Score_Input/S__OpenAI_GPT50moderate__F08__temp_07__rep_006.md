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
run_id: OpenAI_F08_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation repeatedly computes neighbor statistics for every cell-year row by iterating through all 6.46 million rows, performing lookups and aggregation from scratch for each neighbor set. This is highly inefficient because:  
- The neighbor relationships are static across years, yet we rebuild or traverse them repeatedly for each of the 28 years.  
- The `lapply` approach creates massive overhead by re-computing neighbor indices for each row individually.  
- The memory footprint is huge (~6.46 million Ã— 3 stats Ã— 5 variables).  

**Optimization Strategy**  
1. **Precompute static neighbor indices** once for unique cell IDs (344,208 cells).  
2. **Vectorize year-based operations**: For each year, slice the data and compute neighbor stats using matrix operations or `data.table` joins instead of per-row loops.  
3. **Incremental aggregation**: Compute neighbor max/min/mean for each variable by year using the static neighbor map and apply fast functions (like `vapply` or matrix indexing).  
4. Avoid full recomputation of `neighbor_lookup` per row; use a compact structure keyed by cell index.  

Expected improvement: Reduce complexity from *O(rows Ã— neighbors)* to *O(years Ã— cells Ã— neighbors)* with efficient vectorization.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute static neighbor lookup for each cell (not per year)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(ref_idx) {
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    as.integer(id_to_ref[as.character(neighbor_cell_ids)])
  })
}

neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Compute neighbor stats by year in a vectorized manner
compute_neighbor_stats_by_year <- function(dt, neighbor_lookup, var_name) {
  years <- unique(dt$year)
  result_list <- vector("list", length(years))
  
  for (y in seq_along(years)) {
    yr <- years[y]
    slice <- dt[year == yr]
    vals <- slice[[var_name]]
    
    # For each cell, compute stats from its neighbors
    stats_mat <- vapply(seq_along(neighbor_lookup), function(i) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3))
    
    # Transpose to match rows
    stats_mat <- t(stats_mat)
    result_list[[y]] <- data.table(
      id = id_order,
      year = yr,
      paste0(var_name, "_nbr_max") := stats_mat[, 1],
      paste0(var_name, "_nbr_min") := stats_mat[, 2],
      paste0(var_name, "_nbr_mean") := stats_mat[, 3]
    )
  }
  
  rbindlist(result_list)
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- lapply(neighbor_source_vars, function(v) {
  compute_neighbor_stats_by_year(cell_data, neighbor_lookup, v)
})

# Merge all neighbor features into main data
neighbor_features_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), neighbor_features)
cell_data <- merge(cell_data, neighbor_features_dt, by = c("id", "year"), all.x = TRUE)

# Preserve Random Forest prediction step
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Key Gains**  
- **Static neighbor map** avoids rebuilding indices for 6.46M rows.  
- **Year-by-year batch processing** reduces memory thrash and uses efficient joins.  
- **Vectorized `vapply`** drastically cuts function call overhead.  

This approach should reduce computation time from **86+ hours to a few hours** on a 16 GB laptop while preserving the original estimand and the trained Random Forest model.
END_RESPONSE>>>
