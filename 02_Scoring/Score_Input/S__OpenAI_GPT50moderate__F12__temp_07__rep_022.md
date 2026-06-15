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
run_id: OpenAI_F12_T07_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
Your colleagueâ€™s claim that the main bottleneck is `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is incorrect. While `do.call(rbind, ...)` on a 6.46M-element list does incur overhead, the deeper bottleneck lies in **building and repeatedly traversing `neighbor_lookup` as lists of integer vectors combined with multiple `lapply` passes over millions of rows**. Each call to `compute_neighbor_stats()` iterates over all rows (6.46M) for every variable (5 times), repeatedly performing R-level loops and vector filtering (`!is.na(neighbor_vals)`), which is orders of magnitude slower than a vectorized or matrix-based approach.

**Correct Optimization Strategy**  
- Precompute a **dense or sparse neighbor index matrix** where rows correspond to cell-year observations and columns to neighbor indices, avoiding repeated key lookups and string concatenation.
- Store and access values in a **numeric matrix** for all variables instead of lists.
- Use **vectorized operations** (`matrixStats` or `apply`) or `data.table` joins to compute max, min, mean in bulk, minimizing R loops.
- Preserve existing Random Forest model by only transforming feature engineering.

---

### **Optimized Implementation**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Step 1: Build neighbor matrix once
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)),
                         paste(data$id, data$year, sep = "_"))
  
  n <- nrow(data)
  max_nbrs <- max(lengths(neighbors))
  
  # Preallocate a matrix: rows = obs, cols = max_nbrs
  nbr_mat <- matrix(NA_integer_, nrow = n, ncol = max_nbrs)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    keys <- paste(nbr_ids, data$year[i], sep = "_")
    idxs <- idx_lookup[keys]
    # Fill row with neighbor indices
    len <- length(idxs)
    if (len > 0) nbr_mat[i, seq_len(len)] <- idxs
  }
  nbr_mat
}

neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# Step 2: Compute stats in bulk for all variables
compute_neighbor_stats_matrix <- function(data, nbr_mat, var_names) {
  vals_mat <- as.matrix(data[, ..var_names])
  n <- nrow(vals_mat)
  max_nbrs <- ncol(nbr_mat)
  
  # 3D array: (rows x neighbors), compute row-wise ignoring NAs
  result_list <- vector("list", length(var_names))
  
  for (v in seq_along(var_names)) {
    vals <- vals_mat[, v]
    # Gather neighbor values: matrix of size n x max_nbrs
    nbr_vals <- matrix(vals[nbr_mat], nrow = n, ncol = max_nbrs)
    
    # Apply max, min, mean by row, ignoring NAs
    max_vec <- rowMaxs(nbr_vals, na.rm = TRUE)
    min_vec <- rowMins(nbr_vals, na.rm = TRUE)
    mean_vec <- rowMeans(nbr_vals, na.rm = TRUE)
    
    # Replace empty neighbor rows with NA
    no_nbr <- rowSums(!is.na(nbr_vals)) == 0
    max_vec[no_nbr] <- NA
    min_vec[no_nbr] <- NA
    mean_vec[no_nbr] <- NA
    
    result_list[[v]] <- data.table(
      paste0(var_names[v], "_nbr_max") = max_vec,
      paste0(var_names[v], "_nbr_min") = min_vec,
      paste0(var_names[v], "_nbr_mean") = mean_vec
    )
  }
  
  # Combine all new features
  do.call(cbind, result_list)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_neighbor_stats_matrix(cell_data, neighbor_mat, neighbor_source_vars)

# Merge back into cell_data
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Why This is Faster**
- Eliminates repeated R-level loops over 6.46M rows for each variable.
- Switches from list-based lookups to **preallocated matrices**.
- Uses `matrixStats` for **vectorized row operations**, leveraging compiled code.
- Only one pass through data for all 5 variables, instead of 5 full loops.

---

**Expected Performance Improvement**  
This approach reduces runtime from **86+ hours to a few hours or less**, bounded by memory and disk I/O rather than R interpreter overhead. It remains memory-feasible on a 16â€¯GB laptop by processing 5 variables at once and storing neighbor indices compactly.
END_RESPONSE>>>
