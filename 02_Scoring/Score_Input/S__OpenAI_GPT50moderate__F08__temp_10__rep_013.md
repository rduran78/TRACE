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
run_id: OpenAI_F08_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is that computation in `compute_neighbor_stats` loops over ~6.46 million rows for each variable, performing nested lookups into large lists. Since the neighbor structure is static across years, recalculating neighbor indices per row-year results in redundant work. The current O(N Ã— Y Ã— neighbors) approach explodes in size, leading to the 86+ hour estimate and memory pressure.

**Optimization Strategy**  
- Precompute a static neighbor *index matrix* based only on cell ids (size ~344k rows).
- For each year, slice the relevant variable vector, use the static neighbor indices, and compute max/min/mean in a vectorized way.
- Process year-wise in memory-efficient batches rather than across all rows.
- Append features back to the panel after computing year-specific neighbor stats.
- Avoid repeated `lapply` and `do.call` by using matrix operations or vectorized apply.

This reduces complexity from ~6.46M Ã— neighbors per variable repeated for every variable â†’ down to 28 batches with ~344k computations each.

---

### **Optimized Working R Code**

```r
# Build static neighbor matrix once based on cell_id
build_static_neighbor_matrix <- function(id_order, neighbors) {
  # neighbors is spdep::nb for id_order
  max_nbrs <- max(sapply(neighbors, length))
  # Fill matrix with NA for missing slots
  nbr_mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_nbrs)
  for (i in seq_along(neighbors)) {
    if (length(neighbors[[i]]) > 0) {
      nbr_mat[i, seq_along(neighbors[[i]])] <- neighbors[[i]]
    }
  }
  nbr_mat
}

compute_year_neighbor_stats <- function(year_data, nbr_mat, var_name) {
  vals <- year_data[[var_name]]
  # Extract neighbors' values in matrix form
  neighbor_vals <- apply(nbr_mat, 2, function(col_idx) {
    v <- vals[col_idx]
    v[is.na(col_idx)] <- NA_real_
    v
  })
  neighbor_vals <- as.matrix(neighbor_vals)
  # Compute row-wise max, min, mean with NA handling
  max_vals <- apply(neighbor_vals, 1, function(x) if(all(is.na(x))) NA_real_ else max(x, na.rm=TRUE))
  min_vals <- apply(neighbor_vals, 1, function(x) if(all(is.na(x))) NA_real_ else min(x, na.rm=TRUE))
  mean_vals <- apply(neighbor_vals, 1, function(x) if(all(is.na(x))) NA_real_ else mean(x, na.rm=TRUE))
  cbind(max_vals, min_vals, mean_vals)
}

# Main process
optimize_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_vars) {
  nbr_mat <- build_static_neighbor_matrix(id_order, neighbors)
  
  results_list <- vector("list", length(neighbor_vars))
  names(results_list) <- neighbor_vars
  
  # Preallocate for final combination
  cell_data_out <- cell_data
  
  for (yr in sort(unique(cell_data$year))) {
    year_idx <- which(cell_data$year == yr)
    year_data <- cell_data[year_idx, ]
    # Order year_data by id_order if necessary
    year_data <- year_data[match(id_order, year_data$id), ]
    
    # Compute stats for each neighbor var
    for (var_name in neighbor_vars) {
      stats_mat <- compute_year_neighbor_stats(year_data, nbr_mat, var_name)
      colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
      # Merge back by position
      cell_data_out[year_idx, paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))] <- stats_mat
    }
  }
  
  cell_data_out
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

**Why this works**  
- Neighbor lookup (`nbr_mat`) is computed once, size ~344k Ã— max degree.
- Processing is per-year, so only 344k rows in memory at a time. For 28 loop iterations and 5 variables it's practical on 16 GB RAM.
- Feature consistency preserved (max, min, mean).
- No model retraining; we enrich `cell_data` exactly as before.

**Expected performance improvement**  
From 86+ hours â†’ likely under 1-2 hours depending on disk and CPU, due to eliminating 6.46M deep list traversals and redundant concatenations.
END_RESPONSE>>>
