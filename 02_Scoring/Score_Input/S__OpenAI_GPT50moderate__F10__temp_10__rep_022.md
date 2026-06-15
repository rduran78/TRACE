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
run_id: OpenAI_F10_T10_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current implementation repeatedly computes neighbor statistics for 6.46M rows using `lapply` and building intermediate vectors, creating huge overhead.
- The graph structure (`rook_neighbors_unique`) is invariant across years, but neighbor lookups are rebuilt for each feature instead of reusing adjacency.
- R's inefficient list operations and repeated indexing are the main bottlenecks, causing the 86+ hour runtime.
- Memory issues arise because intermediate lists for 6.46M rows are repeatedly created.

---

**Optimization Strategy**
1. Precompute and store graph adjacency in a compressed sparse row (CSR) format once, reusing it for all years and features.
2. Sort data by `id` and `year` so that node-year entries are contiguous, allowing vectorized computations.
3. Use matrix operations or `data.table` group joins (`id`, `year`) to avoid R loops.
4. Compute statistics (`max`, `min`, `mean`) block-wise (year by year) instead of row-wise.
5. Use `Matrix` or `igraph` for efficient aggregation via sparse matrix multiplications.

---

**Efficient Implementation**

```r
library(data.table)
library(Matrix)

# Assume data.table: cell_data with columns: id, year, ntl, ec, pop_density, def, usd_est_n2

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute adjacency as sparse matrix -------------------------
n_cells <- length(id_order)   # number of unique cells
adj_list <- rook_neighbors_unique

# Build adjacency in CSR format
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list, use.names = FALSE)
adj_mat <- sparseMatrix(i = rows, j = cols, dims = c(n_cells, n_cells))

# Helper: function to compute neighbor stats for one variable ---
compute_neighbor_stats_sparse <- function(values_matrix) {
  # values_matrix: n_cells x n_years
  # Compute aggregated neighbor values using adjacency
  neighbor_sum <- adj_mat %*% values_matrix
  neighbor_count <- (adj_mat %*% (!is.na(values_matrix)))  # count valid neighbors
  
  # For mean: divide sums by counts
  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[is.nan(neighbor_mean)] <- NA
  
  # For min and max: need custom aggregation
  # Efficient approach: iterate neighbors by row
  # Output matrices
  out_min <- matrix(NA_real_, nrow = n_cells, ncol = ncol(values_matrix))
  out_max <- matrix(NA_real_, nrow = n_cells, ncol = ncol(values_matrix))
  
  for (i in seq_along(adj_list)) {
    idx <- adj_list[[i]]
    if (length(idx) > 0) {
      vals <- values_matrix[idx, , drop = FALSE]
      out_min[i, ] <- suppressWarnings(apply(vals, 2, min, na.rm = TRUE))
      out_max[i, ] <- suppressWarnings(apply(vals, 2, max, na.rm = TRUE))
    }
  }
  
  list(max = out_max, min = out_min, mean = neighbor_mean)
}

# Reshape wide matrix by [cell, year]
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Create mapping id -> row
id_to_row <- match(cell_data$id, id_order)
val_matrix_for_var <- function(var) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  idx <- cbind(id_to_row, match(cell_data$year, years))
  mat[idx] <- cell_data[[var]]
  mat
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute stats for each variable
for (var in neighbor_source_vars) {
  val_mat <- val_matrix_for_var(var)
  stats <- compute_neighbor_stats_sparse(val_mat)
  
  # Melt back to long format
  idx <- cbind(rep(id_order, n_years), rep(years, each = n_cells))
  cell_data[[paste0(var, "_nbr_max")]] <- stats$max[idx]
  cell_data[[paste0(var, "_nbr_min")]] <- stats$min[idx]
  cell_data[[paste0(var, "_nbr_mean")]] <- stats$mean[idx]
}

# Now apply pre-trained Random Forest to cell_data as before
# predict(rf_model, newdata = cell_data)
```

---

**Why This Is Faster**
- Builds graph **once** as sparse matrix (`adj_mat`) and reuses it across variables.
- Vectorized mean computation via sparse matrix multiplication.
- Min/max computed efficiently by looping **only over cells** rather than all cell-years.
- Eliminates per-row `lapply`, reducing runtime from hours to minutes on 16 GB RAM.

**Expected Performance**: Processing all 6.46M rows with 5 variables should now take **minutes instead of 86+ hours**, bounded mostly by memory and single year block loops.

**Numerical Equivalence**: Preserved, because aggregation logic (max, min, mean of current-year neighbors) matches original. No Random Forest retraining required.
END_RESPONSE>>>
