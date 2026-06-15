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
run_id: OpenAI_F03_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The major bottlenecks are:  
1. **Inefficient neighbor lookup:** `build_neighbor_lookup` uses nested `lapply` with `paste` and repeated lookups for ~6.46M rows, resulting in heavy string operations and object copying.  
2. **Repeated calls in `compute_neighbor_stats`:** Each neighbor feature computation iterates over all rows again, performing redundant indexing (`vals[idx]`) and filtering (`!is.na`) across millions of rows.  
3. **Memory blow-up:** Large lists from `lapply` and `do.call(rbind, â€¦)` over millions of rows cause memory thrashing.  
4. **Prediction loop:** Random Forest inference is likely on a data frame, not a numeric `matrix`, slowing down calculations.  
5. **Model loading:** If the model is reloaded in each iteration, this adds unnecessary I/O overhead.  

---

### **Optimization Strategy**
- **Precompute neighbor index as `integer` vectors (once)** instead of dynamic string keys.
- **Convert data to a `data.table`** for vectorized operations and fast joins.
- **Avoid loops for neighbor stats:** Compute all neighbor features in bulk with pre-mapped indices, store in matrix, and apply `max`, `min`, `mean` efficiently.
- **Reuse neighbor_lookup across all vars.**
- **For prediction:**  
  - Convert predictors to a numeric `matrix` (`data.matrix`).  
  - Use `predict(model, newdata = matrix, type = "response", predict.all = FALSE)`.  
- **Parallelization:** Use `parallel::mclapply` or `future.apply` for multiple vars.

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert cell_data to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor indices (as integers, not strings)
build_neighbor_lookup_fast <- function(cell_data, id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), id_order)
  cell_idx <- seq_len(nrow(cell_data))
  id_year_to_idx <- setNames(cell_idx, paste(cell_data$id, cell_data$year, sep = "_"))

  lapply(cell_idx, function(i) {
    ref_idx <- id_to_idx[[as.character(cell_data$id[i])]]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    nbr_keys <- paste(nbr_ids, cell_data$year[i], sep = "_")
    as.integer(id_year_to_idx[nbr_keys])
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for multiple variables efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  res <- matrix(NA_real_, n, 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) res[i, ] <- c(max(v), min(v), mean(v))
    }
  }
  res
}

# Apply to all neighbor vars in parallel
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

results_list <- mclapply(neighbor_source_vars, function(var) {
  compute_neighbor_stats_fast(cell_data[[var]], neighbor_lookup)
}, mc.cores = detectCores() - 1)

# Bind computed features
for (k in seq_along(neighbor_source_vars)) {
  nm <- neighbor_source_vars[k]
  feats <- results_list[[k]]
  cell_data[, paste0(nm, "_nbr_max") := feats[,1]]
  cell_data[, paste0(nm, "_nbr_min") := feats[,2]]
  cell_data[, paste0(nm, "_nbr_mean") := feats[,3]]
}

# Prepare matrix for Random Forest prediction
predictors <- setdiff(names(cell_data), c("target_var", "id", "year")) # replace target_var
X <- data.matrix(cell_data[, ..predictors])

# Load Random Forest model once
rf_model <- readRDS("rf_model.rds")  # ensure it's pre-trained

# Predict efficiently
cell_data[, prediction := predict(rf_model, newdata = X)]
```

---

### **Expected Impact**
- **Single neighbor lookup build:** O(N) instead of repeated string concatenation.
- **Vectorized feature computation:** Avoids 6.46M Ã— 5 full loops in R, significantly reducing runtime.
- **Prediction on matrix:** Faster inference.
- **Result:** Processing time drops from 86+ hours to a few hours or less, given parallelization and reduced overhead.
END_RESPONSE>>>
