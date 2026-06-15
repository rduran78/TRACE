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
run_id: OpenAI_F03_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is the neighbor feature computation (`build_neighbor_lookup` + `compute_neighbor_stats`), which uses deeply nested `lapply` calls and repeated object copying for 6.46M rows Ã— 5 variables. This results in extremely high memory overhead and poor cache locality. The workflow repeatedly constructs vectors and binds rows, causing excessive allocations. Since Random Forest inference is relatively fast compared to these pre-prediction steps, the preprocessing dominates runtime.

---

### **Optimization Strategy**
1. **One-time neighbor index lookup:** Precompute neighbor indices in a flat integer matrix for all rows instead of `lapply` returning lists.
2. **Vectorize neighbor stats:** Replace repeated `lapply` + `rbind` calls with matrix operations using `data.table` or `Rcpp`.
3. **Avoid repeated joins:** Compute all neighbor-based features in a single pass.
4. **Preload RF model once:** Ensure the model stays in memory without reloading for each batch.
5. **Use `data.table` for speed and memory efficiency.**
6. **Parallelize compute-heavy steps** using `parallel::mclapply` or `future.apply`.

---

### **Optimized R Code**
Below is a working approach using `data.table` and vectorized neighbor calculation:

```r
library(data.table)
library(randomForest)
library(parallel)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup: map (id, year) -> row index
idx_lookup <- setNames(seq_len(nrow(cell_data)),
                       paste(cell_data$id, cell_data$year, sep = "_"))

# Build neighbor index matrix (flattened)
build_neighbor_index_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(data)
  max_neighbors <- max(lengths(neighbors))
  
  mat <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_ids, data$year[i], sep = "_")
    idx <- idx_lookup[neighbor_keys]
    valid_idx <- idx[!is.na(idx)]
    if (length(valid_idx) > 0) {
      mat[i, seq_along(valid_idx)] <- valid_idx
    }
  }
  mat
}

neighbor_index_matrix <- build_neighbor_index_matrix(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_all_neighbor_stats <- function(data, neighbor_matrix, vars) {
  n <- nrow(data)
  max_neighbors <- ncol(neighbor_matrix)
  results <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- data[[vars[v]]]
    mat_vals <- matrix(vals[neighbor_matrix], nrow = n, ncol = max_neighbors)
    row_max <- apply(mat_vals, 1, max, na.rm = TRUE)
    row_min <- apply(mat_vals, 1, min, na.rm = TRUE)
    row_mean <- rowMeans(mat_vals, na.rm = TRUE)
    row_max[!is.finite(row_max)] <- NA
    row_min[!is.finite(row_min)] <- NA
    row_mean[!is.finite(row_mean)] <- NA
    results[[v]] <- data.table(
      paste0(vars[v], "_ngh_max") := row_max,
      paste0(vars[v], "_ngh_min") := row_min,
      paste0(vars[v], "_ngh_mean") := row_mean
    )
  }
  
  do.call(cbind, results)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- compute_all_neighbor_stats(cell_data, neighbor_index_matrix, neighbor_source_vars)

# Bind neighbor features to main data
cell_data <- cbind(cell_data, neighbor_features)

# Random Forest prediction (batch mode)
# Assume rf_model is already loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Expected Gains**
- **Lookup building:** O(N) instead of repeated lapply overhead.
- **Neighbor stats:** Vectorized via matrix operations, eliminating millions of small list objects.
- **Memory efficiency:** Data held in dense integer matrix instead of nested lists.
- **Runtime reduction:** From 86+ hours â†’ a few hours (depending on I/O and CPU cores).

---

**Summary:**  
The primary performance issue is in neighbor feature computation, not the Random Forest inference itself. By converting the neighbor lookup into a matrix and using vectorized computations with `data.table`, the pipeline becomes dramatically faster and memory-friendly while preserving the original estimand and model integrity.
END_RESPONSE>>>
