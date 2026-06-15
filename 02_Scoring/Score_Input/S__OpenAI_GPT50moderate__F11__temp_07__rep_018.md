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
run_id: OpenAI_F11_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is incorrect. Random Forest prediction on ~6.46M rows for 110 variables, even on a laptop, would typically take minutes to a few hours, not 86+ hours. The real bottleneck lies in the neighbor-feature computation for five variables across millions of rows. Specifically:  
- `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over 6.46M elements.  
- Heavy repeated string concatenation and list indexing (`paste`, `idx_lookup[...]`) inside tight loops.  
- Memory churn from repeated `do.call(rbind, ...)`.  

This is an *R-level inefficiency*, not model inference.

---

### **Optimization Strategy:**  
- Precompute the neighbor index lookup as integer vectors instead of concatenated keys.  
- Avoid repeated `paste`/hash lookups by mapping IDs and years to row indices in advance.  
- Use `matrix` operations or `vapply` to reduce overhead.  
- Compute all neighbor stats in one pass if possible, rather than looping over variables.  

---

### **Optimized R Code:**

```r
# Precompute neighbor lookup as integer indices without string ops
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) -> row index
  key_to_idx <- match(
    paste(data$id, data$year, sep = "_"),
    paste(id_order, rep(unique(data$year), each = length(id_order)), sep = "_")
  )

  row_ids <- seq_len(nrow(data))
  years <- data$year
  ids <- data$id
  
  # Vectorized neighbor lookup
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(ids[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_ids, years[i], sep = "_")
    idx <- key_to_idx[match(neighbor_keys, names(key_to_idx))]
    idx[!is.na(idx)]
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_names) {
  vals_list <- lapply(var_names, function(v) data[[v]])
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, nrow = n, ncol = length(var_names) * 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      for (j in seq_along(var_names)) {
        neighbor_vals <- vals_list[[j]][idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          res[i, (j - 1) * 3 + 1] <- max(neighbor_vals)
          res[i, (j - 1) * 3 + 2] <- min(neighbor_vals)
          res[i, (j - 1) * 3 + 3] <- mean(neighbor_vals)
        }
      }
    }
  }
  res
}

# Usage
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, neighbor_vars)
```

---

**Why this works:**  
- Eliminates massive overhead from string concatenation and repeated hashing.  
- Computes stats for all variables in a single loop rather than five separate passes.  
- Reduces R interpreter overhead (millions of function calls â†’ single loop).  

**Expected runtime improvement:** From 86+ hours to a few hours or less (potentially minutes with further vectorization or `data.table` parallelization).  

**Preserves:**  
- Trained Random Forest model (unchanged).  
- Original estimand and logic (neighbor-based max/min/mean).  

**Next steps:** For even faster performance, consider:  
- Converting data to `data.table` and using `parallel::mclapply` for neighbor stats.  
- Precomputing neighbor indices in a compact integer matrix for O(1) access.
END_RESPONSE>>>
