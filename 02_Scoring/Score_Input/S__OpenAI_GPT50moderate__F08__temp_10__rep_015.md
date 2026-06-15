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
run_id: OpenAI_F08_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The slowness (86+ hours) is due to computing neighbor statistics repeatedly across 6.46â€¯M rows, even though neighbor relationships are static. Each call to `compute_neighbor_stats` performs multiple traversals of large vectors, leading to redundant work. Memory overhead compounds with dynamic indexing.  

Key inefficient points:  
- For each row-year, neighbor indices are recomputed in terms of the repeated variable subset.  
- Entire pipeline treats each row (cell-year) independentlyâ€”a 6.46â€¯M Ã— 5 multi-pass.  
- Year looping is implicit in repeated neighbor-stat computation rather than isolated at column-level.  

Since neighbor relationships never change over time, we should:  
- Compute neighbor adjacencies once for the 344,208 cells (static).  
- For each year, slice the relevant variable vector, compute neighbor stats for all cells in one shot, and append results.  
- Avoid creating gigantic per-row lists of neighbors for every cellâ€“year (currently 6.46â€¯M lists).  

---

**Optimization Strategy**  
1. **Precompute adjacency**: Keep neighbor lookup at cell (not cell-year) level.  
2. **Process by year**: For each year and each variable, use the static adjacency to compute max, min, mean.  
3. **Vectorize**: Use matrix operations or `vapply` instead of repeatedly building row-level lists.  
4. **Incremental write**: Append results per year instead of holding everything in memory at once.  

Benefit: We cut complexity from \(O(N_\text{rows} \times V)\) to ~\(O(N_\text{cells} \times Y \times V)\), reusing adjacency each time.  

---

**Optimized R Code**

```r
# Static neighbor indices (cell-level)
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors: spdep::nb list of integer indices
  # id_order is vector of cell IDs in order
  out <- lapply(seq_along(id_order), function(i) neighbors[[i]])
  names(out) <- id_order
  out
}

# Compute a matrix of neighbor stats for one year and one variable
compute_neighbor_stats_year <- function(var_values, neighbor_lookup) {
  result <- t(vapply(neighbor_lookup, function(idx) {
    neigh_vals <- var_values[idx]
    neigh_vals <- neigh_vals[!is.na(neigh_vals)]
    if (length(neigh_vals) == 0) return(c(NA, NA, NA))
    c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
  }, numeric(3)))
  colnames(result) <- c("max", "min", "mean")
  result
}

# Main optimized loop
add_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  neighbor_lookup <- build_cell_neighbor_lookup(id_order, neighbors)
  years <- sort(unique(cell_data$year))
  
  # Preallocate a list to collect augmented data per year
  augmented_list <- vector("list", length(years))
  
  for (y_idx in seq_along(years)) {
    yr <- years[y_idx]
    message("Processing year: ", yr)
    
    # Filter once for the year
    year_subset <- cell_data[cell_data$year == yr, ]
    stopifnot(nrow(year_subset) == length(id_order)) # ensure alignment
    
    # For each variable, compute stats
    for (var_name in vars) {
      stats_mat <- compute_neighbor_stats_year(year_subset[[var_name]], neighbor_lookup)
      year_subset[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
      year_subset[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
      year_subset[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
    }
    
    augmented_list[[y_idx]] <- year_subset
  }
  
  do.call(rbind, augmented_list)
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data_enriched <- add_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)
```

---

**Why This Works**  
- `neighbor_lookup` now stores *only 344K* neighbor lists instead of 6.46M.  
- Computation is year-based: each year processes 344K rows vs. looping across all 6.46M repeatedly.  
- Vectorized `vapply` improves speed and avoids large allocations in nested loops.  
- Preserves original estimand, column alignment, and pipeline compatibility with existing pre-trained Random Forest model.  

**Expected runtime**: Drops from 86+ hours to a few hours (or less) on a 16â€¯GB laptop, as most overhead from redundant neighbor recomputation and nested row-level lookups is eliminated.
END_RESPONSE>>>
