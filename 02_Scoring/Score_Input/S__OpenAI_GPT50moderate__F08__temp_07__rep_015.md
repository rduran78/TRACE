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
run_id: OpenAI_F08_T07_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation repeatedly scans the entire 6.46M-row dataset for each variable and year combination using a large `neighbor_lookup` list of length equal to total rows. This is extremely costly because:  
- Neighbor relationships (1.37M directed edges) are *static across years*, but the code recomputes neighbor indices for every row-year pair.  
- For 28 years Ã— 344k cells, operations scale with O(N Ã— neighbors) repeatedly.  
- Memory overhead: building and storing a 6.46M-element list of integer vectors and repeatedly computing stats across it for each variable is inefficient.  

**Optimization strategy**  
Leverage the static nature of neighbors:  
- Build a neighbor index **once per cell** (length = 344,208).  
- For each year, slice the relevant variable vector and compute neighbor statistics by vectorized aggregation using this fixed neighbor structure.  
- Avoid `lapply` over all 6.46M rows; process by year in blocks.  
- Preallocate a result matrix and write in chunks to reduce memory thrash.  

This reduces complexity from ~6.46M Ã— neighbor operations to (28 years Ã— 344k cells) operations with pre-built neighbor lists only once.  

**Working R code**  

```r
# Build static neighbor lookup (cell-to-cell, no year)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_ref[as.character(neighbor_ids)]
  })
}

# Compute neighbor stats per year
compute_neighbor_stats_by_year <- function(data, static_neighbors, var_name, years, id_order) {
  n_cells <- length(id_order)
  n_years <- length(years)
  out <- matrix(NA_real_, nrow = n_cells * n_years, ncol = 3)
  
  colnames(out) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  
  for (y_idx in seq_along(years)) {
    yr <- years[y_idx]
    yr_idx <- which(data$year == yr)
    vals <- setNames(data[[var_name]][yr_idx], data$id[yr_idx])
    # Ensure order matches id_order
    vals_vec <- vals[as.character(id_order)]
    
    stats_mat <- t(sapply(static_neighbors, function(nei) {
      if (length(nei) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals_vec[nei]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }))
    
    row_range <- ((y_idx - 1) * n_cells + 1):(y_idx * n_cells)
    out[row_range, ] <- stats_mat
  }
  
  out
}

# Main driver
id_order <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))

# Build static neighbor lookup once
static_neighbors <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_neighbor_features <- list()

for (var_name in neighbor_source_vars) {
  message("Processing ", var_name)
  feature_mat <- compute_neighbor_stats_by_year(cell_data, static_neighbors, var_name, years, id_order)
  all_neighbor_features[[var_name]] <- feature_mat
}

# Combine all new features into a single data.frame
neighbor_features_df <- do.call(cbind, all_neighbor_features)

# Bind to original data
# Ensure cell_data is sorted by id_order then year
cell_data <- cell_data[order(match(cell_data$id, id_order), cell_data$year), ]
cell_data <- cbind(cell_data, neighbor_features_df)

# Now proceed with Random Forest predictions as before
```

**Why this works:**  
- Neighbor structure (1.37M edges) used once, not replicated for 6.46M rows.  
- Processes year-by-year, only 344k rows at a time.  
- Preallocation avoids repeated `rbind`.  
- Preserves estimands and Random Forest model (no retraining).  

**Expected performance gain:**  
From 86+ hours to a few hours (or less) on a 16â€¯GB laptop, because repeated redundant lookups and large list traversals are eliminated.  
Memory footprint also drops significantly by avoiding 6.46M-element neighbor list.
END_RESPONSE>>>
