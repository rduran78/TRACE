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
run_id: OpenAI_F03_T07_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is twofold:  
1. **Neighbor feature computation**:  
   - `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M rows, causing excessive R-level overhead and memory allocations.  
   - Frequent creation of intermediate objects (`result`, `paste`, `idx_lookup`) and repeated list-to-matrix binding is costly.  

2. **Prediction loop**:  
   - Predicting row-by-row or in small chunks is slow for Random Forest (especially `randomForest` package in R).  
   - Model loading for each invocation or large-object copying can add overhead.  
   - Memory constraints: Full data + neighbor expansions can exceed 16 GB if not vectorized.  

---

### **Optimization Strategy**

- **Precompute neighbor relationships in integer indices** once and reuse them.  
- Replace `lapply` iteration with **vectorized aggregation** or `data.table` joins.  
- Compute neighbor statistics via **fast grouped operations** rather than row-by-row.  
- Use **matrix-based feature augmentation** instead of repeatedly calling `cbind` or `rbind`.  
- For prediction:  
  - Load the Random Forest model **once**.  
  - Use `predict(model, newdata, type="response")` in **large batches**, not per-row.  
  - Prefer `ranger::predict` for faster inference if possible (supports `randomForest` objects if retraining is not allowed).  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest) # or ranger if model compatible

# 1. Convert cell_data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# 2. Precompute neighbor lookup as a vector of integer vectors
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))

neighbor_lookup <- lapply(seq_along(id_order), function(ref_idx) {
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  paste(neighbor_ids, cell_data$year[1], sep = "_") # placeholder, year will vary later
})

# 3. Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  
  # Preallocate matrix
  res <- matrix(NA_real_, nrow(cell_data), 3L)
  
  for (i in seq_len(nrow(cell_data))) {
    yr <- cell_data$year[i]
    ref_idx <- id_to_idx[[as.character(cell_data$id[i])]]
    neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
    neighbor_keys <- paste(neighbor_ids, yr, sep = "_")
    idx <- idx_lookup[neighbor_keys]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  
  res
}

# 4. Add neighbor features for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(var_name)
  colnames(stats) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}

# 5. Predict in large batches
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)
for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end], type = "response")
}

cell_data[, gdp_pred := preds]
```

---

### **Further Improvements**
- Replace the neighbor-loop logic with `data.table` joins by reshaping neighbor relationships into long format and aggregating by (id, year).  
- If allowed, convert `rf_model` to `ranger` for **10â€“50Ã— faster inference**.  
- Consider compressing numeric input (e.g., `float` via `fst`) to reduce RAM footprint.  

**Expected Impact**:  
- From **86+ hours â†’ several hours** (major speedup from vectorization and batch prediction).  
- Memory stays under 16 GB with careful batching and avoiding repeated copies.
END_RESPONSE>>>
