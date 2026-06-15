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
run_id: OpenAI_F01_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F01_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottlenecks are:  
- Iterating through years in R with `for` + `predict()` on large data frames (hundreds of thousands of rows and many predictors) is slow.  
- Using `dplyr` on such large data can be memory-intensive.  
- `randomForest::predict()` is single-threaded by default, leading to long runtimes.  

---

**Optimization Strategy:**  
1. Use **`data.table`** instead of `dplyr` for filtering and assignmentâ€”faster and memory-efficient.  
2. Enable **parallel prediction** using `foreach` and `parallel` or `future.apply`.  
3. Avoid copying data repeatedly; pre-split test sets by year.  
4. Process in chunks if memory is tight.  

---

**Optimized R Code:**  
```r
library(data.table)
library(parallel)
library(foreach)
library(doParallel)

# Convert to data.table
setDT(pred_db)
setDT(joined_data)

years <- unique(pred_db$year)

# Setup parallel backend
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

pred_db[, consolidated := NA_real_]

# Parallel prediction over years
foreach(yr = years, .packages = c("data.table", "randomForest")) %dopar% {
  message("Predicting for year ", yr)
  
  # Filter rows for this year
  test_set <- joined_data[year == yr]
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    preds <- predict(rf_model, newdata = test_set)
    
    # Assign predictions efficiently
    pred_idx <- which(pred_db$year == yr)
    pred_db$consolidated[pred_idx] <- preds
  } else {
    message("Warning: No model found for year ", yr)
  }
}

stopCluster(cl)

fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Improvements:**  
- Uses `data.table` for fast filtering and assignment.  
- Parallelizes predictions across all available cores.  
- Avoids retraining models, preserves original trained models.  
- Handles large data without unnecessary copies.  

If still memory-constrained, add **chunked prediction** within each year:  
```r
chunk_size <- 50000
for (start in seq(1, nrow(test_set), by = chunk_size)) {
  end <- min(start + chunk_size - 1, nrow(test_set))
  preds[start:end] <- predict(rf_model, newdata = test_set[start:end])
}
```

This ensures scalability on a normal machine.
END_RESPONSE>>>
