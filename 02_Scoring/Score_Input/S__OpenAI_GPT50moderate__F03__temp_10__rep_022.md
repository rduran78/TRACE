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
run_id: OpenAI_F03_T10_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is in feature preparation, particularly `build_neighbor_lookup` and repeated loops for computing neighbor features. Issues:  
- **Excessive object copying** due to repeated `lapply` calls and data frame modifications.  
- **String manipulation overhead** in key creation (`paste`) for every cell-year pair.  
- **Repeated extraction from lists and vectors** causing memory churn.  
- **Non-vectorized neighbor statistics computation** (computes per-row in `lapply`).  
- Large intermediate objects (`neighbor_lookup`) consume RAM and increase GC time.  
Random Forest inference itself is fast; data reshaping dominates time.

---

### **Optimization Strategy**
1. **Precompute numeric indices** instead of string concatenation keys.  
2. Represent `neighbor_lookup` as an `integer` matrix or list optimized for reuse (avoid key lookup in every iteration).  
3. Use **data.table** or **matrix operations** to compute neighbor stats for all rows at once, utilizing vectorization.  
4. Avoid repeated calls to `compute_and_add_neighbor_features`; compute all neighbor stats (max, min, mean) for all variables in a single pass.  
5. Memory tips: do not copy the entire `cell_data` on each loop; instead, `cbind` computed features once.  

---

### **Working Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute numeric key for fast lookup
cell_data[, key := .I]  # row index
id_lookup <- setNames(seq_along(id_order), as.character(id_order))

# Build neighbor lookup as a list of integer indices (fast integer operations)
neighbor_lookup <- lapply(seq_along(id_order), function(ref_idx) {
  idx <- rook_neighbors_unique[[ref_idx]]
  if (length(idx) == 0) return(integer(0))
  # Map neighbor ids to dataset rows (all years)
  neighbor_rows <- cell_data[id %in% id_order[idx], key]
  neighbor_rows
})

# Compute all neighbor statistics in one pass
compute_neighbor_stats_bulk <- function(data, neighbor_lookup, vars) {
  results_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    var_name <- vars[v]
    vals <- data[[var_name]]
    # Preallocate result matrix
    res_mat <- matrix(NA_real_, nrow(data), 3L)
    
    for (i in seq_along(neighbor_lookup)) {
      idx_list <- neighbor_lookup[[i]]
      if (length(idx_list) == 0) next
      # Compute stats across time slice for neighbors
      neighbor_vals <- vals[idx_list]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      res_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
    results_list[[v]] <- res_mat
  }
  results_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_list <- compute_neighbor_stats_bulk(cell_data, neighbor_lookup, neighbor_source_vars)

# Combine computed features efficiently
for (i in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[i]
  mat <- stats_list[[i]]
  cell_data[, paste0(var, "_nbr_max") := mat[,1]]
  cell_data[, paste0(var, "_nbr_min") := mat[,2]]
  cell_data[, paste0(var, "_nbr_mean") := mat[,3]]
}

# Random Forest prediction (preserve trained model)
# Assume rf_model and predictor_vars are preloaded
preds <- predict(rf_model, newdata = cell_data[, ..predictor_vars])
```

---

### **Key Improvements**
- Removed `paste` key concatenation (strings â†’ integers).  
- Eliminated repeated `lapply` in favor of a single bulk computation.  
- All neighbor stats for all variables computed in one loop, reducing overhead by ~70%.  
- Used `data.table` indexing for speed and memory efficiency.  

**Expected runtime**: Hours reduced from 86+ to approximately **2â€“3 hours** on a 16GB laptop, mainly limited by size of data, not algorithmic inefficiency.
END_RESPONSE>>>
