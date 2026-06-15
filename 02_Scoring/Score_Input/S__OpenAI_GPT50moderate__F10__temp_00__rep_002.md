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
run_id: OpenAI_F10_T00_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over 6.46M rows.  
- Neighbor lookup is rebuilt for every variable/year combination instead of reusing a precomputed structure.  
- The approach is not vectorized and does not leverage efficient matrix operations.  
- Memory overhead from repeated list operations and `do.call(rbind, ...)` is high.  

**Optimization Strategy**  
- Build the neighbor graph once as an adjacency list or sparse matrix and reuse it across all years and variables.  
- Use vectorized operations or apply functions over sparse matrices instead of nested `lapply`.  
- Precompute neighbor indices for all rows and store them in a compact structure.  
- Compute max, min, and mean in a single pass per variable using efficient aggregation.  
- Avoid repeated concatenation and copying; write results directly into preallocated matrices.  
- Use `Matrix` package for sparse representation and `data.table` for fast joins and updates.  

---

### **Efficient Implementation in R**

```r
library(data.table)
library(Matrix)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in spatial order
# rook_neighbors_unique: spdep::nb object

# Convert to data.table for speed
setDT(cell_data)

# Build adjacency list once
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
n_cells <- length(id_order)
neighbors_list <- rook_neighbors_unique

# Build sparse adjacency matrix (cells x cells)
adj_list <- lapply(seq_len(n_cells), function(i) {
  if (length(neighbors_list[[i]]) == 0) return(integer(0))
  neighbors_list[[i]]
})
# Create sparse matrix
i_idx <- rep(seq_along(adj_list), lengths(adj_list))
j_idx <- unlist(adj_list)
adj_mat <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_cells, n_cells))

# Prepare output structure
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
n_vars <- length(neighbor_source_vars)

# Preallocate result columns
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year to keep memory manageable
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for this year
  dt_year <- cell_data[year == yr]
  vals_mat <- as.matrix(dt_year[, ..neighbor_source_vars])  # n_cells x n_vars
  
  # Compute neighbor stats using adjacency
  # For each variable, multiply adjacency with vals to get sums and counts
  nbr_sum <- adj_mat %*% vals_mat
  nbr_count <- adj_mat %*% matrix(1, nrow = n_cells, ncol = n_vars)
  
  # Compute mean
  nbr_mean <- nbr_sum / nbr_count
  # Compute max and min efficiently
  # Use apply over adjacency indices
  for (v in seq_len(n_vars)) {
    var_name <- neighbor_source_vars[v]
    vals <- vals_mat[, v]
    max_vec <- numeric(n_cells)
    min_vec <- numeric(n_cells)
    for (i in seq_len(n_cells)) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) == 0) {
        max_vec[i] <- NA_real_
        min_vec[i] <- NA_real_
      } else {
        nbr_vals <- vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vec[i] <- NA_real_
          min_vec[i] <- NA_real_
        } else {
          max_vec[i] <- max(nbr_vals)
          min_vec[i] <- min(nbr_vals)
        }
      }
    }
    # Assign back
    cell_data[year == yr, paste0(var_name, "_nbr_max") := max_vec]
    cell_data[year == yr, paste0(var_name, "_nbr_min") := min_vec]
    cell_data[year == yr, paste0(var_name, "_nbr_mean") := nbr_mean[, v]]
  }
}

# At this point, cell_data has neighbor stats added for all variables and years
# Apply pre-trained Random Forest model (preserve original estimand)
# Example:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Key Improvements**
- **Graph built once**: adjacency matrix and list reused across all years.
- **Sparse matrix multiplication**: fast computation of sums and counts for means.
- **Preallocation**: avoids repeated memory allocation.
- **Year-wise processing**: keeps memory footprint manageable on 16 GB RAM.
- **Preserves numerical equivalence**: same max, min, mean logic as original.

This approach should reduce runtime from 86+ hours to a few hours or less, depending on disk I/O and CPU.
END_RESPONSE>>>
