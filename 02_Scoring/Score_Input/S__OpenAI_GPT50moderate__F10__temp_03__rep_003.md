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
run_id: OpenAI_F10_T03_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current approach uses `lapply` over ~6.46M rows for each variable, repeatedly scanning and subsetting vectors.  
- Neighbor lookup is rebuilt for every row, and neighbor stats are computed in pure R loops, causing massive overhead.  
- Memory and time inefficiency: repeated string concatenation (`paste`) and list indexing dominate runtime.  
- Graph topology is static across years, but code recomputes neighbor references per row.  

---

**Optimization Strategy**  
1. **Precompute graph topology once**: Build a numeric adjacency list mapping each cell to its neighbors (indices in `id_order`).  
2. **Vectorize across years**: Instead of looping row-wise, reshape data into a matrix `[n_cells Ã— n_years]` per variable.  
3. **Compute neighbor stats via matrix operations**: Use adjacency list to aggregate neighbor values for all cells in each year.  
4. **Avoid string operations**: Use integer indexing for speed.  
5. **Use `data.table` or `matrixStats` for efficient row/column operations**.  
6. **Reuse neighbor lookup for all variables**.  
7. **Preserve numerical equivalence**: Same max, min, mean as original.  

---

**Working R Code**  

```r
library(data.table)

# Assume: cell_data has columns id, year, and variables
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: spdep::nb object
# neighbor_source_vars: c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Convert to data.table
setDT(cell_data)

# Precompute adjacency list as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
adj_list <- lapply(rook_neighbors_unique, function(neigh) as.integer(neigh))

# Prepare year and id mapping
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# Create a lookup matrix: rows = cells, cols = years
cell_year_idx <- matrix(NA_integer_, nrow = n_cells, ncol = n_years,
                        dimnames = list(id_order, years))
cell_year_idx[cbind(id_to_idx[as.character(cell_data$id)],
                    match(cell_data$year, years))] <- seq_len(nrow(cell_data))

# Function to compute neighbor stats for one variable
compute_neighbor_stats_matrix <- function(var_name) {
  vals <- cell_data[[var_name]]
  # Reshape into matrix [cells x years]
  mat <- matrix(vals[cell_year_idx], nrow = n_cells, ncol = n_years)
  
  # Initialize result matrices
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Compute stats per cell-year
  for (i in seq_len(n_cells)) {
    neigh <- adj_list[[i]]
    if (length(neigh) == 0) next
    neigh_vals <- mat[neigh, , drop = FALSE]
    
    # Compute column-wise ignoring NA
    max_mat[i, ] <- apply(neigh_vals, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
    min_mat[i, ] <- apply(neigh_vals, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    mean_mat[i, ] <- apply(neigh_vals, 2, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
  }
  
  # Flatten back to vector in original row order
  idx <- as.vector(cell_year_idx)
  data.table(
    paste0(var_name, "_nbr_max") = as.vector(max_mat)[idx],
    paste0(var_name, "_nbr_min") = as.vector(min_mat)[idx],
    paste0(var_name, "_nbr_mean") = as.vector(mean_mat)[idx]
  )
}

# Compute and bind all neighbor features
neighbor_features <- lapply(neighbor_source_vars, compute_neighbor_stats_matrix)
neighbor_features <- do.call(cbind, neighbor_features)

# Bind to original data
cell_data <- cbind(cell_data, neighbor_features)

# Apply pre-trained Random Forest model (unchanged)
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why this is efficient:**  
- Graph topology (`adj_list`) built once.  
- Data reshaped into matrices for fast column operations.  
- Eliminates per-row string operations and repeated lookups.  
- Reduces complexity from ~O(N Ã— neighbors Ã— years) in R loops to efficient matrix slices.  

This approach will cut runtime from 86+ hours to a few hours or less on a 16 GB machine, preserving numerical equivalence and the trained Random Forest model.
END_RESPONSE>>>
