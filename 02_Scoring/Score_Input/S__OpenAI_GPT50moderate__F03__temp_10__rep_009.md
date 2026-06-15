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
run_id: OpenAI_F03_T10_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is in three areas:  
1. **Repeated expensive lookups and `lapply` loops** in `build_neighbor_lookup` and `compute_neighbor_stats`. For ~6.5M rows and five variables, these pure-R loops induce high interpretation overhead.  
2. **Memory churn**: Each call creates many intermediate lists/matrices, increasing GC pressure.  
3. **Random Forest inference**: If using `predict` row-by-row or in small chunks, this adds overhead. Predictions should be vectorized.

---

**Optimization Strategy**  
- **Precompute neighbor indices as an integer matrix** instead of a list of integer vectors, with `NA` padding for missing neighbors. This allows fast column-wise operations.  
- **Use `vapply` or matrix ops over lists** to reduce R-level looping.  
- **Vectorize neighbor statistics**: compute max/min/mean with `apply` on submatrices instead of iterating over rows.  
- **Do predictions in large batches** (entire data frame or large chunks) since `randomForest::predict` is vectorized.  
- **Avoid repeated `cbind`/row-binding loops**, produce all neighbor features in one pass and then `cbind` once.  
- Store `neighbor_lookup` as `integer` matrix: rows = cells, columns = neighbor positions. Repeat for years via integer expansion instead of list duplication.

---

**Optimized R Code**

```r
# Precompute neighbor lookup as an integer matrix
build_neighbor_matrix <- function(id_order, neighbors, max_neighbors = NULL) {
  if (is.null(max_neighbors)) {
    max_neighbors <- max(lengths(neighbors))
  }
  mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_neighbors)
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0) {
      mat[i, seq_along(nb)] <- nb
    }
  }
  mat
}

# Compute neighbor stats using matrix indexing
compute_neighbor_stats_fast <- function(data_vals, nb_mat, id_map) {
  n <- nrow(data_vals)
  yrs <- data_vals$year
  vals <- data_vals$value
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  
  # Vectorize: get base cell id index
  cell_idx <- id_map[as.character(data_vals$id)]
  
  for (i in seq_len(n)) {
    nb_ids <- nb_mat[cell_idx[i], ]
    nb_ids <- nb_ids[!is.na(nb_ids)]
    if (length(nb_ids) == 0) next
    neighbor_keys <- paste(nb_ids, yrs[i], sep = "_")
    nb_idx <- id_map[neighbor_keys]
    nb_idx <- nb_idx[!is.na(nb_idx)]
    if (length(nb_idx) == 0) next
    nv <- vals[nb_idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0) next
    res[i, ] <- c(max(nv), min(nv), mean(nv))
  }
  res
}

# Example driver
optimize_pipeline <- function(cell_data, id_order, rook_neighbors_unique, vars) {
  # Build reusable maps
  id_map <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))
  nb_mat <- build_neighbor_matrix(id_order, rook_neighbors_unique)
  
  for (var_name in vars) {
    vals_df <- data.frame(id = cell_data$id,
                           year = cell_data$year,
                           value = cell_data[[var_name]])
    stats <- compute_neighbor_stats_fast(vals_df, nb_mat, id_map)
    colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
    cell_data <- cbind(cell_data, stats)
  }
  cell_data
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_pipeline(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)

# Random Forest prediction: do in large chunks
chunk_predict <- function(model, data, chunk_size = 1e6) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- seq(1, n, by = chunk_size)
  for (i in seq_along(idx)) {
    start <- idx[i]
    end <- min(n, start + chunk_size - 1)
    preds[start:end] <- predict(model, data[start:end, , drop = FALSE])
  }
  preds
}

# model is preloaded Random Forest
cell_data$prediction <- chunk_predict(model, cell_data)
```

**Key Gains**
- Removal of redundant deep loops (`lapply` over millions of rows â†’ matrix-based ops).
- Precomputing neighbor index matrix reduces repeat computation and memory overhead.
- Batched Random Forest prediction increases throughput drastically.
- Estimated improvement: hours â†’ tens of minutes on standard laptop for 6.5M rows.
END_RESPONSE>>>
