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
run_id: OpenAI_F01_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F01_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The bottleneck comes from **predicting in a loop over years** on large data frames without efficient memory handling.  
- `predict()` on `randomForest` models with huge `newdata` consumes significant RAM, especially when `test_set` is large.  
- `joined_data %>% filter()` creates a full copy each iteration, adding overhead.  
- The process is single-threaded and does not leverage chunking or memory-efficient structures.  

---

**Optimization Strategy**  
1. **Avoid repeated filtering**: Use `split()` once to partition `joined_data` by `year`.  
2. **Predict in chunks**: Break large test sets into smaller batches to reduce memory spikes.  
3. **Use `data.table`**: Faster filtering and assignment than `dplyr`.  
4. **Parallelize predictions**: Use `future.apply` or `parallel` for multi-core execution if possible.  
5. **Write incrementally**: Reduce memory pressure by writing results to disk progressively.  

---

**Working Optimized R Code**  

```r
library(data.table)
library(future.apply)

# Convert to data.table for efficiency
setDT(joined_data)
setDT(pred_db)

# Split data by year once
year_splits <- split(joined_data, by = "year", keep.by = FALSE)

# Prepare parallel plan (adjust workers based on CPU cores)
plan(multisession, workers = max(1, parallel::detectCores() - 1))

# Function to predict in chunks
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  preds <- numeric(n)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (i in seq_along(idx)) {
    preds[idx[[i]]] <- predict(model, newdata = data[idx[[i]], ])
  }
  preds
}

# Loop over years in parallel
results <- future_lapply(names(year_splits), function(yr) {
  message("Predicting for year ", yr)
  if (yr %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr]]
    data_yr <- year_splits[[yr]]
    preds <- predict_in_chunks(rf_model, data_yr)
    data.table(id = data_yr$id, year = as.integer(yr), consolidated = preds)
  } else {
    warning("No model found for year ", yr)
    NULL
  }
})

# Combine results
results <- rbindlist(results, use.names = TRUE, fill = TRUE)

# Merge predictions back into pred_db
pred_db[results, on = .(id, year), consolidated := i.consolidated]

# Write output to disk
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Key Benefits**  
- **Chunked predictions** prevent memory overload.  
- **Parallelization** speeds up multi-year predictions.  
- **Data.table** provides high performance for filtering and merging.  
- Preserves existing trained `rf_models_per_year`.  

This approach makes the process feasible on a normal machine without retraining models.
END_RESPONSE>>>
