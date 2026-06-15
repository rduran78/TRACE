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
run_id: OpenAI_F10_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current code iterates across ~6.46M rows multiple times, performs list-based indexing for each neighbor set, and recomputes stats repeatedly.  
- `neighbor_lookup` replicates neighbor relationships for each row-year combination (6.46M lists), causing memory blow-up and long runtime.  
- Use of `lapply` and `rbind` creates heavy overhead.  
- Graph topology is recomputed conceptually per row instead of leveraging a fixed sparse structure applied across years.  

**Optimization Strategy**  
- Represent neighbor graph once as a sparse adjacency matrix (rows = cells, columns = cells).  
- Use `Matrix` and vectorized operations to compute neighborhood aggregations across all years simultaneously.  
- For each variable, arrange the data as a matrix: rows = cells, columns = years (or flattened as long), then apply adjacency matrix multiplication.  
- Compute `max`, `min`, `mean` using efficient aggregation per node-year from compressed neighbor values without expanding to a list.  
- Work in chunks or matrix form to keep memory manageable and exploit vectorization.  
- Avoid list iteration for 6.46M rowsâ€”replace with sparse linear algebra and fast `rowsum`-style operations.  

---

### **Optimized R Implementation**

```r
library(Matrix)
library(data.table)

# Assume: cell_data with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: cell IDs in same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object
# Preserves RF model predictions, only optimizes feature computation

# ---- Build graph adjacency as sparse matrix ----
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
adj_i <- rep(seq_along(adj_list), sapply(adj_list, length))
adj_j <- unlist(adj_list)
adj_mat <- sparseMatrix(i = adj_i, j = adj_j, x = 1, dims = c(n_cells, n_cells))

# ---- Prepare data as wide matrices for vectorized operations ----
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Map cell IDs to index
id_to_idx <- setNames(seq_along(id_order), id_order)
cell_data[, cell_idx := id_to_idx[as.character(id)]]

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Result storage
result_list <- vector("list", length(neighbor_source_vars))

# ---- Efficient neighbor stats computation ----
for (var_name in neighbor_source_vars) {
  message("Processing ", var_name)
  # Convert to cell x year matrix
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(cell_data$cell_idx, match(cell_data$year, years))] <- cell_data[[var_name]]
  
  # Compute mean: (A %*% values) / neighbor_count
  neighbor_counts <- rowSums(adj_mat)
  mean_mat <- (adj_mat %*% mat) / ifelse(neighbor_counts == 0, NA_real_, neighbor_counts)
  
  # Compute max and min per cell-year: need iterative since no direct sparse op
  # We'll extract neighbor indices once
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- adj_list[[i]]
    if (length(nbrs) > 0) {
      vals <- mat[nbrs, , drop = FALSE]
      max_mat[i, ] <- apply(vals, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
      min_mat[i, ] <- apply(vals, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    }
  }
  
  # Reshape back to long
  df <- data.table(
    id = rep(id_order, times = n_years),
    year = rep(years, each = n_cells),
    paste0(var_name, "_nbr_max") = as.vector(max_mat),
    paste0(var_name, "_nbr_min") = as.vector(min_mat),
    paste0(var_name, "_nbr_mean") = as.vector(mean_mat)
  )
  
  result_list[[var_name]] <- df
}

# Merge all neighbor stats back into main data
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), result_list)
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"))

# ---- Apply pre-trained Random Forest model ----
# Assuming model object is `rf_model`
# preds <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Fast**
- Graph topology built once as `adj_mat` (sparse, memory-efficient).
- For `mean`, uses one sparse matrix multiplication for all years and nodes.
- For `max`/`min`, iterates over cells **only once** per var and computes all years in a vectorized block (much faster than per-row loops).
- Avoids creating 6.46M lists and redundant lookups.
- Memory footprint scales with `n_cells Ã— n_years Ã— number_of_vars`, not `n_rows Ã— neighbors`.

**Performance Expectation:**  
Instead of 86+ hours, likely under 1â€“2 hours on 16â€¯GB RAM for 344k cells Ã— 28 years Ã— 5 vars, given sparse ops and vectorization.

**Numerical Equivalence:**  
The logic reproduces the original max, min, and mean computations exactly for each cell-year neighbor set. The RF model remains unchanged.
END_RESPONSE>>>
