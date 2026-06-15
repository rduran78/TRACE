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
run_id: OpenAI_F07_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the `compute_neighbor_stats` loop (and `compute_and_add_neighbor_features`) operating in pure R over ~6.46M rows and millions of neighbor lookups. The nested `lapply` for every variable introduces redundant passes over the same neighbor structure and incurs huge R-level overhead. Building repeated vectors and row-bind calls on such scale explodes memory/time cost.  

Main inefficiencies:  
- Many small lists â†’ repeated allocations.  
- Looping through `neighbor_source_vars` and recomputing stats each time.  
- Not leveraging vectorized or compiled operations (C++/data.table).  

---

**Optimization Strategy**  
1. Compute all neighbor stats in **one pass** rather than separately per variable.  
2. Use `data.table` for fast column access and binding.  
3. Pre-build neighbor index matrix for efficient access.  
4. Use `vapply` or vectorized methods over lists to reduce interpreter overhead.  
5. Optionally go to C++ via `Rcpp` for maximal speed if still too slow.  

Design goals:  
- Keep exact rook-neighbor relationships and same numerical estimand.  
- Avoid retraining model (so preserve feature names and pipeline compatibility).  
- Memory-aware: avoid creating huge intermediate lists repeatedly.  

---

**Optimized Implementation (R with data.table)**  

```r
library(data.table)

# Assume cell_data is already a data.table
setDT(cell_data)

# Build neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor_lookup into a list of integer indices (already returns that)

# Vectorized computation for multiple variables in one pass:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  # Preallocate a matrix: nrow = nrow(data), ncol = length(vars) * 3
  n <- nrow(data)
  k <- length(vars)
  res_mat <- matrix(NA_real_, n, k * 3)
  colnames(res_mat) <- as.vector(t(outer(vars, c("_max", "_min", "_mean"), paste0)))
  
  # For faster access, convert to list of numeric vectors
  var_lists <- lapply(vars, function(v) data[[v]])
  
  # Iterate rows (fast lapply, but do numeric operations in Rcpp for real speed)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(vars)) {
      vals <- var_lists[[j]][idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        res_mat[i, (j - 1) * 3 + 1] <- max(vals)
        res_mat[i, (j - 1) * 3 + 2] <- min(vals)
        res_mat[i, (j - 1) * 3 + 3] <- mean(vals)
      }
    }
  }
  
  res_mat
}

# Apply and cbind efficiently
res <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)
cell_data <- cbind(cell_data, as.data.table(res))
```

---

**Further Speed Gains**  
- Replace inner loops with an `Rcpp` function operating on integer neighbor lists and numeric vectors (orders of magnitude faster).  
- If memory allows, store neighbor_lookup as an integer matrix (ragged list flattened with offsets).  
- Consider parallelization with `future.apply` or `parallel::mclapply` if CPU cores available.  

**Expected Improvement**: Processing 6.46M rows with combined vectorization and compiled code should drop runtime from 86+ hours to well under 1â€“2 hours on a 16GB laptop.  

This preserves the trained Random Forest model, keeps identical estimand, and ensures integrity of rook-neighbor topology.
END_RESPONSE>>>
