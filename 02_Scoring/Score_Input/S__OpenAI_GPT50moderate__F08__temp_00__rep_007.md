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
run_id: OpenAI_F08_T00_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M rows) by iterating through a large list of neighbor indices. This results in repeated lookups and redundant computations across years, even though the neighbor structure is static. The complexity is roughly `O(N * k)` per variable, where `N` is 6.46M and `k` is average neighbor count, multiplied by 5 variables. This is memory- and CPU-intensive for a 16 GB laptop.

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute neighbor relationships once at the cell level (344,208 cells), not per cell-year.
- **Vectorize across years**: For each variable, reshape data into a matrix of size `cells Ã— years` and compute neighbor stats using matrix operations.
- **Avoid repeated list traversals**: Use `rowsum` or `apply` on matrices instead of looping through 6.46M rows.
- **Memory efficiency**: Work in chunks if needed, but 344k Ã— 28 matrix (~9.6M entries) is manageable in memory.

**Working R Code**

```r
# Precompute neighbor lookup at cell level (static)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_ref[as.character(neighbor_ids)])
  })
}

# Compute neighbor stats using matrix operations
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
# Assume cell_data has columns: id, year, and variables
id_order <- sort(unique(cell_data$id))
year_order <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(year_order)

# Reshape data into wide matrix per variable
reshape_to_matrix <- function(data, var_name, id_order, year_order) {
  mat <- matrix(NA_real_, nrow = length(id_order), ncol = length(year_order),
                dimnames = list(id_order, year_order))
  idx <- match(paste(data$id, data$year), 
               paste(rep(id_order, each = n_years), rep(year_order, times = n_cells)))
  mat[idx] <- data[[var_name]]
  mat
}

neighbor_lookup <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge back
for (var_name in neighbor_source_vars) {
  var_matrix <- reshape_to_matrix(cell_data, var_name, id_order, year_order)
  stats <- compute_neighbor_stats_matrix(var_matrix, neighbor_lookup)
  
  # Melt back to long format
  long_idx <- match(paste(cell_data$id, cell_data$year),
                    paste(rep(id_order, each = n_years), rep(year_order, times = n_cells)))
  
  cell_data[[paste0(var_name, "_nbr_max")]]  <- stats$max[long_idx]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- stats$min[long_idx]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats$mean[long_idx]
}

# Now cell_data has neighbor features ready for Random Forest prediction
```

**Why this works**  
- Reduces complexity from 6.46M row-wise loops to 344k cell-wise loops with vectorized year operations.
- Handles all years in one pass per variable.
- Preserves original estimand and Random Forest model.
- Fits in memory: 344k Ã— 28 â‰ˆ 9.6M doubles (~77 MB per matrix).

**Expected runtime**: From 86+ hours to a few minutes on a standard laptop.
END_RESPONSE>>>
