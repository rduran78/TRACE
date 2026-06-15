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
run_id: OpenAI_F10_T10_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current implementation recomputes neighbor indices and calculates statistics for each row and variable using `lapply` loops, causing severe overhead (86+ hours for ~6.46M rows).
- `neighbor_lookup` is list-based and recalculated repeatedly for each variable aggregation step.
- The approach is not vectorized and doesnâ€™t exploit graph structure for shared computations across years and variables.
- Data replication across years makes neighbor statistics redundant since topology is static, yet recomputation is repeated.

---

**Optimization Strategy**  
1. **Precompute Graph Topology Once:** Build a single adjacency list or edge index keyed by cell IDs; reuse for all years.
2. **Use Matrix Operations:** Store data as a numeric matrix for fast indexed access rather than iterative `lapply`.
3. **Sparse Matrix Aggregation:** Represent neighbor relationships as a sparse adjacency matrix and use matrix multiplication for min, max, and mean (with care for NA handling).
4. **Batch Computation Across Years/Variables:** Compute stats for all rows of a variable in one go, not per row.
5. **Memory Management:** Convert `cell_data` to `data.table` for efficient join and in-place updates.
6. **Preserve Numerical Equivalence:** Handle NAs consistently and ensure aggregation logic matches reference.

---

**Efficient R Implementation**  

```r
library(data.table)
library(Matrix)

# Assume cell_data: data.table(id, year, ntl, ec, pop_density, def, usd_est_n2)
# id_order: vector of unique ids in correct order
# rook_neighbors_unique: list of neighbors

# Precompute adjacency as sparse dgCMatrix (rows and cols = ids in id_order)
n_ids <- length(id_order)
id_index <- setNames(seq_along(id_order), id_order)

i_idx <- unlist(lapply(seq_along(rook_neighbors_unique), function(i) rep(i, length(rook_neighbors_unique[[i]]))))
j_idx <- unlist(rook_neighbors_unique)
adj <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_ids, n_ids))

# Convert to data.table and index
setDT(cell_data)
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_matrix <- function(var_values_matrix, adj) {
  # var_values_matrix: n_ids x n_years
  # Compute stats across neighbors using sparse aggregations
  max_fun <- function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
  min_fun <- function(x) if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE)
  mean_fun <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
  
  # Use apply over rows after multiplying adjacency
  apply_stats <- function(xmat, fun) {
    apply(xmat, 2, function(col) { # col = years vector
      res <- pmax.col(cbind(col), na.last = NA) # placeholder, vectorized fallback
    })
  }
  
  # Vectorized alternative: gather indices per node from adjacency in rows
  idx_list <- split(j_idx, i_idx) # neighbors per node
  n_years <- ncol(var_values_matrix)
  res_max <- matrix(NA_real_, n_ids, n_years)
  res_min <- matrix(NA_real_, n_ids, n_years)
  res_mean <- matrix(NA_real_, n_ids, n_years)
  
  for (i in seq_len(n_ids)) {
    nbr <- idx_list[[i]]
    if (length(nbr) == 0) next
    vals <- var_values_matrix[nbr, , drop = FALSE]
    res_max[i, ] <- apply(vals, 2, max_fun)
    res_min[i, ] <- apply(vals, 2, min_fun)
    res_mean[i, ] <- apply(vals, 2, mean_fun)
  }
  
  list(max = res_max, min = res_min, mean = res_mean)
}

# Prepare wide matrix by id x year for each var
years <- sort(unique(cell_data$year))
n_years <- length(years)

for (var_name in neighbor_source_vars) {
  mat <- matrix(NA_real_, nrow = n_ids, ncol = n_years)
  # Fill matrix
  for (i in seq_len(n_years)) {
    yr <- years[i]
    vals <- cell_data[year == yr, ..var_name][[1]]
    mat[ , i] <- vals
  }
  
  stats <- compute_neighbor_stats_matrix(mat, adj)
  
  # Melt back and cbind to cell_data
  max_vec <- as.vector(t(stats$max))
  min_vec <- as.vector(t(stats$min))
  mean_vec <- as.vector(t(stats$mean))
  
  cell_data[, paste0(var_name, "_nbr_max") := max_vec]
  cell_data[, paste0(var_name, "_nbr_min") := min_vec]
  cell_data[, paste0(var_name, "_nbr_mean") := mean_vec]
}

# Apply pre-trained Random Forest model without retraining
# rf_model: pre-trained model
preds <- predict(rf_model, newdata = cell_data)
```

---

**Why Itâ€™s Fast:**  
- Topology computed once as a sparse matrix.
- Batch processing avoids per-row `lapply`.
- Matrix-based aggregation drastically reduces overhead from millions of tiny operations.
- Data.table operations keep join and assignment efficient.

This cuts runtime from 86+ hours to a few hours or less depending on I/O and RAM constraints, while preserving numerical equivalence.
END_RESPONSE>>>
