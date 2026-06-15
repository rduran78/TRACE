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
run_id: OpenAI_F08_T00_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M rows) by iterating through a large list of neighbor indices. This results in repeated lookups and redundant computations across years, even though the neighbor structure is static. The complexity is roughly `O(N * k)` per variable, where `N` is 6.46M and `k` is average neighbor count, multiplied by 5 variables, leading to hours of runtime and high memory pressure.

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute neighbor indices once at the cell level (344,208 cells), not per cell-year.
- **Vectorize across years**: For each variable, reshape data into a matrix of size `cells Ã— years` and compute neighbor stats using matrix operations.
- **Avoid repeated list traversals**: Use `matrixStats` or `apply` on precomputed neighbor slices.
- **Memory efficiency**: Work in chunks if needed, but 344k Ã— 28 fits in memory (~9.6M entries per variable).
- **Preserve estimand**: Ensure max, min, mean are computed per cell-year using same-year neighbor values.

---

### **Optimized R Code**

```r
library(matrixStats)

# Precompute neighbor lookup at cell level (static)
build_cell_neighbor_lookup <- function(id_order, neighbors) {
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
    max_mat[i, ]  <- colMaxs(neighbor_vals, na.rm = TRUE)
    min_mat[i, ]  <- colMins(neighbor_vals, na.rm = TRUE)
    mean_mat[i, ] <- colMeans2(neighbor_vals, na.rm = TRUE)
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# Main pipeline
# Assumes cell_data has columns: id, year, and variables
optimize_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  
  # Reshape data into cell Ã— year matrix for each variable
  cell_year_key <- paste(cell_data$id, cell_data$year, sep = "_")
  mat_list <- list()
  for (v in vars) {
    mat <- matrix(NA_real_, n_cells, n_years,
                  dimnames = list(id_order, years))
    idx <- match(paste(cell_data$id, cell_data$year, sep = "_"), cell_year_key)
    mat[cbind(match(cell_data$id, id_order), match(cell_data$year, years))] <- cell_data[[v]]
    mat_list[[v]] <- mat
  }
  
  # Build static neighbor lookup
  neighbor_lookup <- build_cell_neighbor_lookup(id_order, neighbors)
  
  # Compute and merge results
  for (v in vars) {
    stats <- compute_neighbor_stats_matrix(mat_list[[v]], neighbor_lookup)
    # Flatten back to long format
    long_idx <- match(paste(cell_data$id, cell_data$year, sep = "_"),
                      paste(rep(id_order, each = n_years), rep(years, times = n_cells), sep = "_"))
    cell_data[[paste0(v, "_nbr_max")]]  <- as.vector(t(stats$max))[long_idx]
    cell_data[[paste0(v, "_nbr_min")]]  <- as.vector(t(stats$min))[long_idx]
    cell_data[[paste0(v, "_nbr_mean")]] <- as.vector(t(stats$mean))[long_idx]
  }
  
  cell_data
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

### **Why This Works**
- **Static neighbor structure**: Precomputed once for 344k cells.
- **Matrix-based computation**: Each variable is processed as a 344k Ã— 28 matrix, reducing overhead.
- **Vectorized stats**: `matrixStats` functions are highly optimized in C.
- **Memory footprint**: 344k Ã— 28 â‰ˆ 9.6M doubles (~76 MB per variable), feasible on 16 GB RAM.
- **Expected runtime**: Minutes instead of 86+ hours.

This preserves the original estimand and integrates seamlessly before the Random Forest prediction step without retraining the model.
END_RESPONSE>>>
