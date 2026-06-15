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
run_id: OpenAI_F01_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F01_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is the loop predicting year by year on large data frames with `dplyr` filtering and base R assignment. Both operations are slow and memory-heavy for hundreds of thousands of rows per year. Predictions themselves are fast, but repetitive subsetting and copying of large data frames cause inefficiency.

---

**Optimization Strategy:**  
1. Avoid repeated `dplyr` filtering and base assignment inside the loopâ€”switch to `data.table` for in-place operations.
2. Pre-split or index by year to reduce overhead.
3. Use `predict(..., newdata, type="response")` in vectorized chunks if memory is limited.
4. Ensure factor levels in `test_set` match the model (already assumed).
5. Write output incrementally or after processing to avoid memory duplication.

---

**Optimized R Code:**
```r
library(data.table)

# Convert to data.table for efficient filtering and assignment
setDT(pred_db)
setDT(joined_data)

# Create an index for faster subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# Loop over models efficiently
for (yr in names(rf_models_per_year)) {
  cat("Predicting for year", yr, "\n")
  
  # Subset using data.table (no copy)
  test_set <- joined_data[J(as.integer(yr))]  # or as.numeric if needed
  
  if (nrow(test_set) > 0) {
    rf_model <- rf_models_per_year[[yr]]
    
    # Predict in chunks to reduce memory load
    chunk_size <- 50000
    preds <- vector("numeric", nrow(test_set))
    
    for (i in seq(1, nrow(test_set), by = chunk_size)) {
      idx <- i:min(i + chunk_size - 1, nrow(test_set))
      preds[idx] <- predict(rf_model, newdata = test_set[idx])
    }
    
    # Assign predictions in place
    pred_db[J(as.integer(yr)), consolidated := preds]
  } else {
    cat("Warning: No data for year", yr, "\n")
  }
}

# Save output
fwrite(pred_db, "RF_imputated_db.csv")
```

---

**Why this helps:**  
- `data.table` avoids repeated copies and is far faster for filtering and assignment.
- Chunked prediction prevents exhausting RAM when `newdata` is large.
- Keys (`setkey`) make joins and lookups O(1) per group.
- Preserves trained models and only performs predictions.  

This approach should make the process feasible on a personal computer.
END_RESPONSE>>>
