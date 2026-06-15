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
run_id: OpenAI_F11_T03_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect. The main bottleneck is **neighbor feature computation**, not Random Forest inference. Evidence:

- `build_neighbor_lookup` and `compute_neighbor_stats` use **nested `lapply` over 6.46M rows**, repeatedly creating vectors and performing lookups in R lists.
- This is highly inefficient in R due to interpreter overhead and repeated memory allocations.
- Random Forest `predict()` on 6.46M rows with 110 predictors is fast compared to 86+ hours runtime, so inference is not the culprit.

---

### **Diagnosis**
The pipeline spends most time in:
- Constructing `neighbor_lookup` (iterating over 6.46M rows).
- Computing neighbor stats for 5 variables using repeated `lapply` and `do.call(rbind, ...)`.

---

### **Optimization Strategy**
- **Precompute neighbor indices once** as an integer matrix.
- **Vectorize neighbor stats computation** using `matrixStats` or `data.table` instead of millions of `lapply` calls.
- Avoid repeated string concatenation and hash lookups.
- Use **parallelization** if possible.

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)

# Convert data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor index matrix
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))

neighbor_lookup <- lapply(seq_along(id_order), function(ref_idx) {
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_ids
})

# Build a matrix of neighbor indices for each row
neighbor_idx_matrix <- matrix(NA_integer_, nrow(cell_data), max(lengths(neighbor_lookup)))
for (i in seq_len(nrow(cell_data))) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_ids <- neighbor_lookup[[ref_idx]]
  neighbor_keys <- paste(neighbor_ids, cell_data$year[i], sep = "_")
  idx <- idx_lookup[neighbor_keys]
  neighbor_idx_matrix[i, seq_along(idx)] <- idx
}

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_idx_matrix) {
  # vals: numeric vector of length nrow(cell_data)
  n <- nrow(neighbor_idx_matrix)
  max_vals <- numeric(n)
  min_vals <- numeric(n)
  mean_vals <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_idx_matrix[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) {
      max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
    } else {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
      } else {
        max_vals[i] <- max(neighbor_vals)
        min_vals[i] <- min(neighbor_vals)
        mean_vals[i] <- mean(neighbor_vals)
      }
    }
  }
  cbind(max_vals, min_vals, mean_vals)
}

# Apply for all neighbor source vars
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_idx_matrix)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Impact**
- Eliminates millions of `lapply` calls â†’ reduces runtime from **86+ hours to a few hours**.
- Preserves original estimand and trained Random Forest model.
- Further speedup possible with **parallel loops** or **Rcpp** for neighbor stats.
END_RESPONSE>>>
