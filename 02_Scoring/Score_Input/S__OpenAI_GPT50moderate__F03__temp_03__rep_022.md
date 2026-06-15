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
run_id: OpenAI_F03_T03_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive `lapply` loops** over ~6.46M rows for neighbor feature computation.  
2. **Inefficient object copying** when repeatedly modifying `cell_data`.  
3. **Random Forest prediction overhead** from row-by-row or chunked prediction rather than vectorized/batched prediction.  
4. **Memory pressure** from large intermediate lists and repeated `do.call(rbind, ...)`.  

**Optimization Strategy**  
- Precompute neighbor indices as an integer matrix or list once and reuse.  
- Replace `lapply` with **vectorized operations** or `data.table` for speed and memory efficiency.  
- Compute all neighbor stats for all variables in a **single pass** rather than looping per variable.  
- Use `data.table` keyed joins instead of repeated `paste` string lookups.  
- Perform Random Forest predictions in **large batches** (e.g., 100k rows) to avoid overhead.  
- Avoid modifying `cell_data` repeatedly; build a single feature matrix and bind once.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup as integer indices (list of integer vectors)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  names(vals_list) <- vars
  
  # Preallocate result matrix
  res_mat <- matrix(NA_real_, nrow = nrow(data), ncol = length(vars) * 3)
  colnames(res_mat) <- as.vector(t(outer(vars, c("_max", "_min", "_mean"), paste0)))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(vars)) {
      vals <- vals_list[[j]][idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        res_mat[i, (j - 1) * 3 + 1:3] <- c(max(vals), min(vals), mean(vals))
      }
    }
  }
  res_mat
}

neighbor_features <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind neighbor features to cell_data
cell_data <- cbind(cell_data, neighbor_features)

# Prepare feature matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target_var
X <- as.data.frame(cell_data[, ..predictors])

# Load trained Random Forest model
rf_model <- readRDS("trained_rf_model.rds")

# Batched prediction to avoid memory issues
batch_size <- 100000
n <- nrow(X)
pred <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  pred[start:end] <- predict(rf_model, X[start:end, , drop = FALSE])
}

cell_data$prediction <- pred
```

---

### **Expected Gains**
- Eliminates repeated string operations and multiple passes over 6.46M rows.
- Reduces memory churn by preallocating and batching.
- Moves from 86+ hours to **a few hours or less** on a 16 GB laptop.

**Key Principles:**  
- Use `data.table` for large data manipulation.  
- Preallocate and compute in a single pass.  
- Batch predictions for Random Forest.  

This preserves the trained model and the original estimand while drastically improving efficiency.
END_RESPONSE>>>
