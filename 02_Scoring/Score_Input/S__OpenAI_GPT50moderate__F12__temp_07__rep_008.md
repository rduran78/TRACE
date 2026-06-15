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
run_id: OpenAI_F12_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. Binding a few million rows after computing statistics is relatively cheap compared to the cost of repeatedly scanning neighbor indexes and subsetting vectors inside the `lapply` for 6.46M rows. The **true bottleneck** is the repeated R-level iteration and vector subsetting (`vals[idx]`) inside `compute_neighbor_stats()`. Each iteration incurs overhead in R's interpreter and repeated memory allocations. With ~6.46M rows Ã— 5 variables, this dominates runtime.

---

### **Optimization Strategy**
- Precompute neighbor indices once (already done via `neighbor_lookup`).
- Replace the R `lapply` loop with a **vectorized or compiled approach** (e.g., `data.table`, `Rcpp`, or `vapply` with preallocated matrix).
- Compute all neighbor stats in **C-level loops** via `Rcpp` for speed.
- Avoid repeated copying of `vals[idx]`; instead, iterate over numeric vectors directly in compiled code.

---

### **Optimized Rcpp Implementation**

**Step 1: Install Rcpp if needed**
```r
install.packages("Rcpp")
```

**Step 2: Implement C++ function**
```r
library(Rcpp)

cppFunction('
NumericMatrix computeNeighborStatsCpp(List neighbor_lookup, NumericVector vals) {
  int n = neighbor_lookup.size();
  NumericMatrix result(n, 3); // cols: max, min, mean

  for (int i = 0; i < n; i++) {
    IntegerVector idx = neighbor_lookup[i];
    int m = idx.size();

    double maxv = R_NegInf;
    double minv = R_PosInf;
    double sumv = 0.0;
    int count = 0;

    for (int j = 0; j < m; j++) {
      int pos = idx[j] - 1; // R is 1-based
      if (pos >= 0 && pos < vals.size()) {
        double v = vals[pos];
        if (!R_IsNA(v)) {
          if (v > maxv) maxv = v;
          if (v < minv) minv = v;
          sumv += v;
          count++;
        }
      }
    }

    if (count == 0) {
      result(i, 0) = NA_REAL;
      result(i, 1) = NA_REAL;
      result(i, 2) = NA_REAL;
    } else {
      result(i, 0) = maxv;
      result(i, 1) = minv;
      result(i, 2) = sumv / count;
    }
  }
  return result;
}
')
```

---

### **Step 3: Replace compute_neighbor_stats**
```r
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  computeNeighborStatsCpp(neighbor_lookup, vals)
}
```

---

### **Step 4: Apply in loop**
```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Performance Gain**
- Eliminates 6.46M R-level loops â†’ replaced by a single efficient C++ loop.
- Drastically reduces overhead from function calls and memory allocations.
- Brings computation down from **86+ hours to a few minutes** on a standard laptop.

---

âœ… **Summary:** Reject colleagueâ€™s diagnosis. The real bottleneck is R-level per-row neighbor aggregation inside `compute_neighbor_stats()`, not `do.call(rbind, ...)`. Correct fix is to move that logic into a compiled routine (Rcpp) for efficient looping and aggregation while preserving the original estimand and trained model.
END_RESPONSE>>>
