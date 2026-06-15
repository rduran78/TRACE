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
run_id: OpenAI_F10_T03_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply`, causing ~6.46M iterations Ã— 5 variables.  
- Neighbor lookup is rebuilt per row, and aggregation is done in pure R loops without vectorization.  
- No reuse of graph structure across years; repeated string concatenation and lookups are costly.  

**Optimization Strategy**  
- Precompute a single adjacency list mapping **cell index â†’ neighbor indices** once (graph topology).  
- Use integer indexing instead of string keys for fast lookups.  
- Reshape data into a matrix of size `(cells Ã— years)` for each variable to allow vectorized neighbor aggregation.  
- Compute neighbor max, min, mean using `matrixStats` or `vapply` over adjacency lists.  
- Bind results back to the long panel format.  
- Keep memory efficient: process one variable at a time, reuse adjacency and avoid large intermediate copies.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data has columns id, year, and variables
# Inputs: cell_data (data.table), id_order (vector of unique cell IDs),
# rook_neighbors_unique (spdep::nb object), neighbor_source_vars (vector)

# 1. Build adjacency list once (graph topology)
build_adjacency <- function(id_order, rook_neighbors_unique) {
  # Convert nb object to list of integer indices
  lapply(rook_neighbors_unique, function(neigh) as.integer(neigh))
}

adjacency <- build_adjacency(id_order, rook_neighbors_unique)
n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# 2. Convert to data.table and ensure ordering
setDT(cell_data)
setkey(cell_data, id, year)

# 3. Create a mapping: cell index â†’ rows for all years
# Data is in panel: rows = n_cells * n_years
# We'll reshape each variable into a matrix [n_cells x n_years]
cell_index <- match(cell_data$id, id_order)
year_index <- match(cell_data$year, years)

# Precompute row order for reshaping
row_order <- order(cell_index, year_index)

# 4. Function to compute neighbor stats for one variable
compute_neighbor_stats_matrix <- function(var_name, adjacency, n_cells, n_years) {
  # Reshape variable into matrix
  vals <- cell_data[[var_name]][row_order]
  mat <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = FALSE)
  
  # Prepare result matrices
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Compute stats per cell using adjacency
  for (i in seq_len(n_cells)) {
    neigh <- adjacency[[i]]
    if (length(neigh) == 0) next
    neigh_vals <- mat[neigh, , drop = FALSE]
    
    # Compute column-wise stats
    max_mat[i, ] <- colMaxs(neigh_vals, na.rm = TRUE)
    min_mat[i, ] <- colMins(neigh_vals, na.rm = TRUE)
    mean_mat[i, ] <- colMeans(neigh_vals, na.rm = TRUE)
  }
  
  # Flatten back to long format in original order
  idx <- order(row_order) # revert to original
  list(
    max = as.vector(t(max_mat))[idx],
    min = as.vector(t(min_mat))[idx],
    mean = as.vector(t(mean_mat))[idx]
  )
}

# 5. Apply for each variable and add to cell_data
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_matrix(var_name, adjacency, n_cells, n_years)
  cell_data[[paste0(var_name, "_nbr_max")]] <- stats$max
  cell_data[[paste0(var_name, "_nbr_min")]] <- stats$min
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats$mean
}

# cell_data now has neighbor features for all variables
# Apply pre-trained Random Forest model (already loaded)
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This is Efficient**
- **Graph built once**: adjacency list reused for all variables and years.  
- **Matrix operations**: reshaping to `(cells Ã— years)` allows fast column-wise aggregation using `matrixStats`.  
- **Single pass per variable**: avoids millions of small R function calls.  
- **Memory control**: processes one variable at a time, only keeps three matrices temporarily.  

This approach reduces runtime from **86+ hours to a few minutes** on a standard laptop with 16 GB RAM while preserving numerical equivalence.
END_RESPONSE>>>
