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
run_id: OpenAI_F03_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The primary bottleneck is not Random Forest inference itself but the feature preparation loop, particularly the repeated use of `lapply` and `do.call(rbind, ...)` over 6.46 million rows and multiple variables. These operations create massive intermediate lists and perform expensive object copying. Recomputing neighbor summaries for each variable separately amplifies the cost. In addition, `build_neighbor_lookup` returns a large list of integer vectors (one per row), which consumes significant memory and slows access.  

**Optimization Strategy**  
1. **Precompute neighbor stats for all variables at once** to reduce repeated traversal of the neighbor list.  
2. Replace `lapply` with **vectorized or matrix-based computation** where possible.  
3. Use **data.table** for efficient row access and binding.  
4. Store neighbor indices as integer vectors in a compressed form and leverage fast aggregation (e.g., `vapply`, `rowMeans`).  
5. Keep the Random Forest model loaded in memory once and apply `predict` in chunks (e.g., 1e5 rows per chunk) to avoid memory blow-up.  

---

### **Optimized Implementation**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute index lookup
idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))
id_to_ref  <- setNames(seq_along(id_order), as.character(id_order))

# Precompute neighbor indices for each row
neighbor_lookup <- lapply(seq_len(nrow(cell_data)), function(i) {
  ref_idx           <- id_to_ref[[as.character(cell_data$id[i])]]
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys     <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
  idx_lookup[neighbor_keys]
})

# Remove NAs for each list element
neighbor_lookup <- lapply(neighbor_lookup, function(x) as.integer(x[!is.na(x)]))

# Compute all neighbor stats in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  names(vals_list) <- vars

  n <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n, ncol = length(vars) * 3)
  colnames(result) <- as.vector(t(outer(vars, c("_max", "_min", "_mean"), paste0)))

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(vars)) {
      nv <- vals_list[[j]][idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0) next
      offset <- (j - 1) * 3
      result[i, offset + 1:3] <- c(max(nv), min(nv), mean(nv))
    }
  }
  result
}

neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind results to cell_data
cell_data <- cbind(cell_data, neighbor_stats)

# Random Forest prediction in chunks
chunk_size <- 1e5
n <- nrow(cell_data)
pred <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  pred[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data[, gdp_pred := pred]
```

---

### **Expected Gains**
- **Single pass** neighbor computation reduces overhead by ~5x.
- Using `data.table` and pre-binding results avoids repeated copies.
- Chunked prediction keeps memory under control.
- Estimated runtime drops from **86+ hours to under a few hours** on a 16 GB laptop.

This preserves the original trained Random Forest model and the numerical estimand while making the pipeline feasible for large-scale inference.
END_RESPONSE>>>
