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
run_id: OpenAI_F09_T10_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow because it recomputes neighbor relationships for every variable across all ~6.46M rows. The `build_neighbor_lookup` and `compute_neighbor_stats` functions process lists of length `nrow(data)` repeatedly. This massive repetition for five variables over 6.46M rows causes the 86+ hours runtime. Moreover, using `lapply` row-by-row is highly inefficient.  

**Optimization Strategy**  
1. **Build neighbor lookup once**: Map each row to its neighbor row indices just once (already done in `build_neighbor_lookup`).
2. **Precompute all neighbor stats in a single pass**: Instead of looping over variables and running `lapply` repeatedly, iterate once through all rows and calculate max, min, and mean for all neighbor variables together.
3. **Use matrix operations**: Store `vals` for the five variables in a numeric matrix; do vectorized calculations where possible.
4. **Avoid repeated list processing**: Convert the neighbor lookup to an integer vector-list once, then reuse.
5. **Preserve RF model and estimand**: Only optimize feature computation.

**Optimized R Code**  

```r
# Assume cell_data, id_order, rook_neighbors_unique available.

# Step 1: Build neighbor lookup once
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))

  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}

neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Step 2: Compute all neighbor stats for selected variables in one pass
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  vals_mat <- as.matrix(data[, var_names])
  n <- nrow(vals_mat)
  p <- length(var_names)
  
  # Preallocate result matrix: 3 stats per var
  res <- matrix(NA_real_, nrow = n, ncol = p * 3)
  colnames(res) <- as.vector(t(outer(var_names, c("_nb_max", "_nb_min", "_nb_mean"), paste0)))
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    nb_vals <- vals_mat[idx, , drop = FALSE]
    nb_vals <- nb_vals[is.finite(nb_vals[,1]), , drop = FALSE] # Filter rows with NAs
    if (nrow(nb_vals) == 0) next
    res[i, seq(1, by = 3, length.out = p)]     <- apply(nb_vals, 2, max, na.rm = TRUE)
    res[i, seq(2, by = 3, length.out = p)]     <- apply(nb_vals, 2, min, na.rm = TRUE)
    res[i, seq(3, by = 3, length.out = p)]     <- apply(nb_vals, 2, mean, na.rm = TRUE)
  }
  
  as.data.frame(res)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Combine with original data
cell_data <- cbind(cell_data, neighbor_stats)

# Proceed with RF prediction using pre-trained model
# predictions <- predict(rf_model, newdata = cell_data)
```

**Why this is faster**  
- Neighbor lookup built once (O(N)).
- All five variables computed in a single loop over rows (O(N * average_neighbors)) instead of 5 separate loops.
- Results stored in one preallocated matrix, reducing overhead.
- Takes advantage of vectorized `apply` instead of repeated `lapply`.

**Expected Impact**  
This should reduce runtime from 86+ hours to a few hours or less, depending on I/O and CPU efficiency, making it feasible on a 16GB RAM laptop.
END_RESPONSE>>>
