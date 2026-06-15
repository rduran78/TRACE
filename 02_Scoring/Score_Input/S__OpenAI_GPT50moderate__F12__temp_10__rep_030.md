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
run_id: OpenAI_F12_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The primary bottleneck is **not** `do.call(rbind, result)` or list binding. The much deeper bottleneck lies in **repeated R-level loops over 6.46M panel rows across 5 variables**, where each row iterates over its neighbors, performs character pasting, and indices into `vals`. This is \( O(N \times k) \) (â‰ˆ 6.46M Ã— 4â€“8 neighbors) across millions of calls, dominated by interpretation overhead and redundant lookups in pure R.

### **Diagnosis**
- `build_neighbor_lookup()` already precomputes neighbor indices onceâ€”good.
- In `compute_neighbor_stats()`, *every call for every var_name re-traverses the full 6.46M row structure in R using `lapply`*, doing per-row function calls which crush performance.
- `do.call(rbind, result)` is additive, but negligible compared to ~32M nested operations.

### **Correct Optimization**
Compute all neighbor statistics in **vectorized or compiled form** rather than per-row R loops:
- Precompute `neighbor_lookup` **once** (we keep this).
- For each variable, avoid millions of R closures; use `vapply` or collapse with `data.table` / matrix ops.
- Best: Convert long format to indexed vectors and use `vapply` or `cppFunction` (Rcpp) for efficient aggregation.

Below is a pure R vectorized solution using `vapply` + preallocated matrix:

---

### **Optimized Code**

```r
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, n, 3L, dimnames = list(NULL, c("max", "min", "mean")))

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx)) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals)) {
        res[i, 1] <- max(neighbor_vals)
        res[i, 2] <- min(neighbor_vals)
        res[i, 3] <- mean(neighbor_vals)
      }
    }
  }
  res
}

# Or parallelize across variables:
library(data.table)
setDT(cell_data)  # convert to data.table for speed

neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  set(cell_data, j = paste0(var_name, "_nbr_max"), value = stats[, 1])
  set(cell_data, j = paste0(var_name, "_nbr_min"), value = stats[, 2])
  set(cell_data, j = paste0(var_name, "_nbr_mean"), value = stats[, 3])
}
```

---

### **Why It Works**
- Eliminates repeated function closures and `lapply` + `do.call` overhead.
- Pre-allocates final matrix; no list binding.
- Operates in tight for-loop at C-level (`[`) rather than millions of lapply closures.
- Memory footprint is constant.

On a standard laptop, this drops runtime from **86+ hours to a few hours**, and if combined with `parallel::mclapply` or Rcpp, can fall under 1 hour.

**Preserves**:
- Original estimands (exact same neighbor max/min/mean).
- Trained Random Forest model remains untouched.
END_RESPONSE>>>
