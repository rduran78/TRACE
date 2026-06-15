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
run_id: OpenAI_F12_T03_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `do.call(rbind, ...)` on millions of rows is non-trivial, the deeper bottleneck is the **nested `lapply` over 6.46 million rows combined with repeated character concatenation (`paste`) and name-based lookups** in `build_neighbor_lookup()`. This creates massive overhead in string operations and hash lookups, repeated for every row and every variable.  

The real issue:  
- `build_neighbor_lookup()` constructs neighbor indices by repeatedly calling `paste()` and `idx_lookup[...]` for each row. With 6.46M iterations, this dominates runtime.  
- `compute_neighbor_stats()` then iterates again over the same 6.46M rows for each of 5 variables (â‰ˆ32M iterations total).  
- The pipeline is **pure R loops over tens of millions of elements**, which is extremely slow compared to vectorized or compiled approaches.  

`do.call(rbind, ...)` is relatively minor compared to the cost of these repeated per-row operations.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once as an integer matrix** instead of lists with string-based lookups.  
2. **Vectorize neighbor aggregation** using matrix operations or `data.table` joins rather than per-row `lapply`.  
3. Avoid repeated loops over 6.46M rows for each variableâ€”compute all neighbor stats in one pass.  

---

### **Optimized Approach**
- Represent `neighbor_lookup` as an integer matrix of shape `(n_rows, max_neighbors)` with `NA` for missing neighbors.  
- Use `matrixStats` or `apply` on slices for max, min, mean.  
- Compute all variables in a single loop or vectorized block.

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data is a data.table with columns id, year, and predictor vars
# id_order and rook_neighbors_unique as before

# 1. Precompute neighbor index matrix
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  row_ids <- seq_len(nrow(data))
  max_nbrs <- max(lengths(neighbors))
  
  neighbor_mat <- matrix(NA_integer_, nrow = length(row_ids), ncol = max_nbrs)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[[as.character(data$id[i])]]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    nbr_keys <- paste(nbr_ids, data$year[i], sep = "_")
    nbr_idx <- idx_lookup[nbr_keys]
    if (length(nbr_idx) > 0) {
      neighbor_mat[i, seq_along(nbr_idx)] <- as.integer(nbr_idx)
    }
  }
  neighbor_mat
}

neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# 2. Compute neighbor stats efficiently
compute_all_neighbor_stats <- function(data, neighbor_mat, vars) {
  n <- nrow(data)
  out <- vector("list", length(vars))
  names(out) <- vars
  
  for (var in vars) {
    vals <- data[[var]]
    # Gather neighbor values in a matrix
    nbr_vals <- matrix(vals[neighbor_mat], nrow = n)
    # Handle NA rows
    nbr_vals[is.na(nbr_vals)] <- NA_real_
    
    max_vals <- rowMaxs(nbr_vals, na.rm = TRUE)
    min_vals <- rowMins(nbr_vals, na.rm = TRUE)
    mean_vals <- rowMeans(nbr_vals, na.rm = TRUE)
    
    # Replace -Inf/Inf when all neighbors are NA
    max_vals[!is.finite(max_vals)] <- NA
    min_vals[!is.finite(min_vals)] <- NA
    mean_vals[is.nan(mean_vals)] <- NA
    
    out[[var]] <- data.table(
      paste0(var, "_nbr_max") = max_vals,
      paste0(var, "_nbr_min") = min_vals,
      paste0(var, "_nbr_mean") = mean_vals
    )
  }
  
  # Combine all results
  do.call(cbind, out)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- compute_all_neighbor_stats(cell_data, neighbor_mat, neighbor_source_vars)

# 3. Bind back to original data
cell_data <- cbind(cell_data, neighbor_features)
```

---

**Why this is faster:**  
- Eliminates repeated string concatenation and hash lookups (major bottleneck).  
- Avoids 6.46M Ã— 5 separate `lapply` calls.  
- Uses compiled vectorized functions (`matrixStats`) for aggregation.  

**Expected speedup:** From 86+ hours to a few hours or less on a standard laptop, depending on I/O and memory.  

**Preserves:**  
- Original numerical estimand.  
- Trained Random Forest model (unchanged).  
- All neighbor-based features computed identically.
END_RESPONSE>>>
