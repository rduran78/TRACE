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
run_id: OpenAI_F12_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding are the main bottlenecks is incorrect. While `rbind` on a large list is non-trivial, the **true bottleneck** lies in `compute_neighbor_stats()`â€™s inner loop, which repeatedly **extracts neighbor indices and subsets large vectors millions of times inside `lapply`**, causing heavy R-level overhead and cache inefficiency. This dominates runtime far more than the final `rbind`.

Given:
- ~6.46M rows and 5 variables â†’ ~32.3M neighbor summaries computed.
- Each call: vector subset + NA filtering + `max/min/mean` â†’ billions of interpreted operations.

The issue is algorithmic: computing neighbor stats row-by-row in R is extremely slow. The fix is to **vectorize and shift work to compiled code (matrix ops)**, avoiding millions of R function calls.

---

### **Optimization Strategy**
1. Precompute a **neighbor index matrix** for all rows (fixed across vars).
2. Use **matrix/vectorized operations** to calculate `max`, `min`, and `mean` for each rowâ€™s neighbors in bulk.
3. Avoid R loops: move from `lapply` to matrix-based summaries.
4. Keep RAM footprint manageable by processing in chunks if needed.

---

### **Optimized Approach**
- Build a fixed-size integer matrix `neighbor_mat` (rows = cell-year rows, cols = max neighbor count).
- For each variable:
  - Extract its full column as numeric vector.
  - Index neighbors in one shot: `vals_mat <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))`.
  - Apply `rowMaxs`, `rowMins`, and `rowMeans` (from **matrixStats**), handling NAs efficiently.
- Append results to `cell_data`.

---

### **Working R Code**

```r
library(matrixStats)

# Precompute neighbor matrix with NA fill for unequal lengths
build_neighbor_matrix <- function(neighbor_lookup) {
  max_n <- max(lengths(neighbor_lookup))
  n_rows <- length(neighbor_lookup)
  neighbor_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_n)
  for (i in seq_len(n_rows)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx)) neighbor_mat[i, seq_along(idx)] <- idx
  }
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  vals_mat <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))
  # Compute row-wise stats ignoring NAs
  row_max <- rowMaxs(vals_mat, na.rm = TRUE)
  row_min <- rowMins(vals_mat, na.rm = TRUE)
  row_mean <- rowMeans(vals_mat, na.rm = TRUE)
  # Replace NaN (all neighbors NA) with NA_real_
  row_mean[is.nan(row_mean)] <- NA_real_
  cbind(row_max, row_min, row_mean)
}

# ---- Main Pipeline ----
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_mat <- build_neighbor_matrix(neighbor_lookup)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Fix Works**
- Eliminates **millions of R function calls** â†’ replaced by compiled operations.
- Avoids repeated neighbor indexing (`vals[idx]` inside loops).
- Memory-efficient: integer matrix of neighbors + on-the-fly slices.
- Preserves estimands and Random Forest model (only feature engineering changes).
- Expected runtime improvement: **from 86+ hours â†’ under 1 hour** on 16 GB laptop.

**Bottom line:** The bottleneck is R-level iteration and indexing, not `rbind`. The correct optimization is full vectorization with `matrixStats`.
END_RESPONSE>>>
