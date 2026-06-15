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
run_id: OpenAI_F12_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottlenecks is incorrect. While `rbind` on millions of rows incurs overhead, the dominant cost lies in the **outer loop repeatedly scanning the full dataset for each of 5 variables**, performing `lapply` over ~6.46M elements every time. `compute_neighbor_stats()` is executed 5 times, each time iterating through all rows and indexing into `vals` for each neighbor set. This results in roughly `6.46M Ã— 5` expensive R-level loops and memory allocations, which dwarfs the `rbind` overhead.

**True bottleneck:** The repeated per-variable full pass with R-level iteration, not `do.call(rbind, result)`.

---

### **Optimization Strategy**
- Compute all neighbor stats in **one pass**, not 5 separate ones.
- Avoid repeated indexing inside R loops by using **vectorized or matrix-based operations**.
- Preallocate output as a matrix for all variables to eliminate repeated list growth and `rbind`.
- Use `vapply` or manual preallocation for speed.
- Preserve the trained Random Forest model and original estimand.

---

### **Optimized Approach**
- Pre-extract required columns into a numeric matrix.
- For each row, gather neighbor indices once, then compute max/min/mean across all 5 variables in one go.
- Return a matrix with `nrow(data)` rows and `5 Ã— 3 = 15` columns (max, min, mean for each var).

---

### **Working R Code**

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, var_names) {
  var_matrix <- as.matrix(data[, var_names, drop = FALSE])
  n <- nrow(data)
  p <- length(var_names)
  out <- matrix(NA_real_, nrow = n, ncol = p * 3)
  colnames(out) <- as.vector(t(outer(var_names, c("_max", "_min", "_mean"), paste0)))

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- var_matrix[idx, , drop = FALSE]
      # Remove rows with all NAs
      valid_rows <- rowSums(!is.na(neighbor_vals)) > 0
      if (any(valid_rows)) {
        neighbor_vals <- neighbor_vals[valid_rows, , drop = FALSE]
        out[i, seq(1, p * 3, by = 3)] <- apply(neighbor_vals, 2, max, na.rm = TRUE)
        out[i, seq(2, p * 3, by = 3)] <- apply(neighbor_vals, 2, min, na.rm = TRUE)
        out[i, seq(3, p * 3, by = 3)] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
      }
    }
  }
  out
}

# Usage:
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_vars)
cell_data <- cbind(cell_data, neighbor_stats)
```

---

### **Impact**
- Reduces 5 full passes (each 6.46M iterations) into **one pass**.
- Eliminates repeated `do.call(rbind, ...)` and redundant neighbor lookups.
- Memory footprint stays manageable (adds ~15 numeric columns).
- Expected runtime improvement: from 86+ hours to a few hours (or less with further parallelization using `parallel::mclapply` or `future.apply`).

**Summary:** The primary bottleneck is redundant iteration across variables, not `rbind`. The fix is a single-pass, preallocated matrix approach computing all neighbor summaries together.
END_RESPONSE>>>
