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
run_id: OpenAI_F03_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- **Main bottleneck:** The neighbor feature computation uses `lapply` repeatedly for millions of rows, creating large intermediate lists and performing repeated object copying (`cell_data` updates per variable).  
- **Prediction loop inefficiency:** Repeated transformations before Random Forest inference amplify runtime.  
- **Memory pressure:** 6.46M rows Ã— 110 predictors + 15 neighbor-derived features is huge; list-based operations and repeated `rbind` calls are expensive.  
- **Model loading is minor compared to feature generation overhead.** Random Forest inference (`predict` on millions of rows) is slower but not the main cause of 86 hoursâ€”the neighbor stats preparation dominates.

---

**Optimization Strategy**  
1. **Vectorize neighbor feature computation:**  
   - Replace `lapply` loops with `vapply` or matrix operations where possible.  
   - Preallocate matrices for neighbor statistics rather than incremental `rbind`.  
2. **Avoid repeated data frame copying:**  
   - Compute all neighbor stats first and `cbind` once.  
3. **Use integer indices and avoid repeated `paste` calls inside loops:**  
   - Build `neighbor_lookup` as integer index lists; reuse efficiently.  
4. **Parallelize computation:**  
   - Use `parallel::mclapply` or `future.apply` for multi-core execution.  
5. **Prediction optimization:**  
   - Use `predict(..., newdata, type="response")` in chunks (e.g., 500k rows) to manage memory and speed.  
6. **Keep model unchanged; preserve estimand.**

---

**Optimized Working R Code**

```r
library(data.table)
library(parallel)

# Assume cell_data is a data.table for efficiency
setDT(cell_data)

# Build neighbor lookup (vectorized, no repeated paste)
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  keys <- paste(rep(id_order, each = length(unique(data$year))),
                rep(unique(data$year), times = length(id_order)), sep = "_")
  
  # Precompute for efficiency
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    as.integer(idx_lookup[neighbor_keys[neighbor_keys %in% names(idx_lookup)]])
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats in parallel
compute_neighbor_stats_fast <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  n <- nrow(data)
  res_mat <- matrix(NA_real_, n, length(vars) * 3)
  
  # Parallel loop
  mclapply(seq_len(n), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(rep(NA_real_, length(vars) * 3))
    out <- numeric(length(vars) * 3)
    for (j in seq_along(vars)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        out[(j-1)*3 + 1:3] <- NA_real_
      } else {
        out[(j-1)*3 + 1:3] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
    out
  }, mc.cores = detectCores() - 1) -> rows
  
  res_mat[] <- do.call(rbind, rows)
  colnames(res_mat) <- as.vector(sapply(vars, function(v) paste0(v, c("_max","_min","_mean"))))
  res_mat
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind features once
cell_data <- cbind(cell_data, neighbor_features)

# Random Forest prediction in chunks
chunk_size <- 5e5
n <- nrow(cell_data)
preds <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end], type = "response")
}

cell_data$pred_gdp <- preds
```

---

**Expected Impact:**  
- Reduces neighbor computation from hours to minutes with parallelization and preallocation.  
- Avoids repeated copying and inefficient loops.  
- Scales inference with chunking to fit 16 GB RAM.  
- Preserves trained Random Forest model and original estimand.
END_RESPONSE>>>
