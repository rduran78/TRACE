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
run_id: OpenAI_F08_T00_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M rows) by iterating through a large list of neighbor indices. This is inefficient because:  
- The neighbor structure is static across years, but the code redundantly rebuilds neighbor relationships for every row.  
- `compute_neighbor_stats` repeatedly scans large vectors and performs many small list operations.  
- The process is not vectorized and does not leverage the panel structure (28 years Ã— 344,208 cells).  

**Optimization Strategy**  
- Precompute a static neighbor index for cells only (not cell-years).  
- Reshape data into a matrix of size `n_cells Ã— n_years` for each variable.  
- Compute neighbor max, min, and mean using matrix operations for each year, leveraging the static neighbor list.  
- Bind results back to the long panel efficiently.  
- Avoid loops over 6.46M rows; instead, loop over 28 years and 5 variables (much smaller).  

This reduces complexity from O(n_rows Ã— avg_neighbors) to O(n_years Ã— n_cells Ã— avg_neighbors), which is far more manageable in memory and time.  

---

### **Optimized R Code**

```r
# Precompute static neighbor lookup for cells
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_ref[as.character(neighbor_ids)]
  })
}

# Compute neighbor stats for each variable using matrix operations
compute_neighbor_stats_matrix <- function(var_matrix, neighbor_lookup) {
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)
  
  max_mat <- matrix(NA_real_, n_cells, n_years)
  min_mat <- matrix(NA_real_, n_cells, n_years)
  mean_mat <- matrix(NA_real_, n_cells, n_years)
  
  for (i in seq_len(n_cells)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- var_matrix[idx, , drop = FALSE]
    max_mat[i, ]  <- apply(neighbor_vals, 2, max, na.rm = TRUE)
    min_mat[i, ]  <- apply(neighbor_vals, 2, min, na.rm = TRUE)
    mean_mat[i, ] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# Main pipeline
optimize_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  # Ensure data is sorted by id, then year
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]
  
  # Build static neighbor lookup
  neighbor_lookup <- build_static_neighbor_lookup(id_order, neighbors)
  
  n_cells <- length(id_order)
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  # Create a mapping from (id, year) to row index
  row_index <- match(paste(cell_data$id, cell_data$year),
                     outer(id_order, years, paste, sep = "_"))
  
  # For each variable, compute neighbor stats
  for (var_name in vars) {
    # Reshape into matrix: rows = cells, cols = years
    var_matrix <- matrix(cell_data[[var_name]], nrow = n_cells, ncol = n_years, byrow = FALSE)
    
    stats <- compute_neighbor_stats_matrix(var_matrix, neighbor_lookup)
    
    # Flatten back to long format
    cell_data[[paste0(var_name, "_nbr_max")]]  <- as.vector(stats$max)[row_index]
    cell_data[[paste0(var_name, "_nbr_min")]]  <- as.vector(stats$min)[row_index]
    cell_data[[paste0(var_name, "_nbr_mean")]] <- as.vector(stats$mean)[row_index]
  }
  
  cell_data
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

**Why This Works**  
- **Static neighbor structure**: We compute neighbors once for 344,208 cells, not 6.46M rows.  
- **Matrix operations**: For each variable, we handle 344,208 Ã— 28 matrix instead of 6.46M rows individually.  
- **Memory efficiency**: Each matrix is about 344,208 Ã— 28 â‰ˆ 9.6M elements (~80 MB per variable), which fits in 16 GB RAM.  
- **Time efficiency**: Loops over 344k cells Ã— 28 years Ã— 5 variables, but each iteration uses fast `apply` on small slices.  

**Preserves**  
- Original numerical estimand (neighbor max, min, mean).  
- Pre-trained Random Forest model (no retraining).  

**Expected Runtime**  
From 86+ hours to a few hours (or less) on a standard laptop.
END_RESPONSE>>>
