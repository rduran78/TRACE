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
run_id: OpenAI_F12_T03_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottlenecks is incorrect. While `do.call(rbind, ...)` adds overhead, the deeper bottleneck is the repeated use of `lapply` over **6.46 million rows** combined with per-row vector operations and repeated neighbor lookups. This results in massive R-level loop overhead and memory churn. The true performance killer is the row-wise computation in `compute_neighbor_stats()` and the inefficient structure of `neighbor_lookup` as a list of varying-length integer vectors.

---

**Optimization Strategy:**  
- Avoid per-row `lapply` and repeated subsetting.
- Precompute a **long-format neighbor table** (cell-year â†’ neighbor-year) and join it to the data.
- Use **vectorized aggregation** via `data.table` or `dplyr` instead of millions of R-level loops.
- Preserve the trained Random Forest model and numerical estimand by ensuring identical max/min/mean computations.

---

### **Optimized Approach Using `data.table`**

```r
library(data.table)

# Convert to data.table for fast joins and aggregation
setDT(cell_data)

# Build long-format neighbor table once
build_neighbor_table <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  # Expand all cell-year rows with their neighbors
  res_list <- vector("list", length = nrow(data))
  for (i in seq_len(nrow(data))) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    neighbor_idx <- idx_lookup[neighbor_keys]
    neighbor_idx <- neighbor_idx[!is.na(neighbor_idx)]
    if (length(neighbor_idx) > 0) {
      res_list[[i]] <- data.table(source_idx = i, neighbor_idx = neighbor_idx)
    }
  }
  rbindlist(res_list)
}

neighbor_table <- build_neighbor_table(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for all variables in one pass
compute_neighbor_features <- function(data, neighbor_table, vars) {
  # Map neighbor indices to actual values
  neighbor_table[, source_id := data$id[source_idx]]
  neighbor_table[, year := data$year[source_idx]]
  
  # Join neighbor values
  neighbor_table[, (vars) := data[neighbor_idx, ..vars]]
  
  # Aggregate by source_idx
  agg <- neighbor_table[, lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(x), min(x), mean(x))
  }), by = source_idx, .SDcols = vars]
  
  # Split columns into separate features
  feature_cols <- list()
  for (v in vars) {
    feature_cols[[paste0(v, "_nb_max")]] <- sapply(agg[[v]], `[`, 1)
    feature_cols[[paste0(v, "_nb_min")]] <- sapply(agg[[v]], `[`, 2)
    feature_cols[[paste0(v, "_nb_mean")]] <- sapply(agg[[v]], `[`, 3)
  }
  
  # Bind back to original data
  data[agg$source_idx, names(feature_cols) := feature_cols]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
compute_neighbor_features(cell_data, neighbor_table, neighbor_source_vars)
```

---

### **Why This Is Faster**
- Eliminates 6.46M row-wise `lapply` calls.
- Performs aggregation in **compiled C-level loops** via `data.table`.
- Reduces memory churn by using a single long-format table instead of millions of small lists.
- Preserves original estimand (max, min, mean of non-NA neighbors).

**Expected Impact:**  
This approach should cut runtime from **86+ hours to a few hours or less** on a standard laptop, given efficient joins and aggregation.

**Summary:**  
Reject the colleagueâ€™s diagnosis. The bottleneck is the massive row-wise R-level computation, not `do.call(rbind)`. The correct optimization is to restructure the pipeline into a vectorized, table-based join and aggregate workflow.
END_RESPONSE>>>
