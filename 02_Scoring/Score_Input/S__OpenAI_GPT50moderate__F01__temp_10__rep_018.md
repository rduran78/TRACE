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
run_id: OpenAI_F01_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F01_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is the **for-loop iterating over years** and filtering `joined_data` for each year combined with predictions on very large data frames. Each iteration copies and subsets large objects, which is memory and time intensive. Predicting one year at a time using `rf_models_per_year` is correct but inefficient.  

---

### **Optimization Strategy**
1. **Vectorize and Minimize Copying:**  
   - Avoid repeated subsetting with `dplyr` inside a loop. Instead, use `data.table` for fast filtering and in-place updates.
2. **Chunk-Based Prediction:**  
   - Predict in chunks (`split` or data.table batching) to avoid loading entire data into memory at once.
3. **Parallelization:**  
   - Use `parallel` or `foreach` with appropriate backend for predictions.
4. **Preserve Models:**  
   - Don't retrain; reuse `rf_models_per_year`.
5. **I/O Optimization:**  
   - Write output once at the end, not inside the loop.

---

### **Optimized Working Code**

```r
library(data.table)
library(parallel)

# Convert data frames to data.table for speed
setDT(pred_db)
setDT(joined_data)

# Ensure rf_models_per_year exists from loaded RData
# Detect available cores
num_cores <- max(1, detectCores() - 1)

# Function for prediction per year
predict_year <- function(year) {
  if (as.character(year) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(year)]]
    test_set <- joined_data[year == !!year]
    preds <- predict(rf_model, newdata = test_set)
    return(data.table(cell_id = test_set$cell_id, year = year, preds = preds))
  } else {
    warning(paste0("No model found for year ", year))
    return(NULL)
  }
}

# Parallel apply over years
years <- unique(pred_db$year)
res_list <- mclapply(years, predict_year, mc.cores = num_cores)

# Combine all predictions
pred_all <- rbindlist(res_list, use.names = TRUE, fill = TRUE)

# Merge predictions back into pred_db by cell_id and year
setkey(pred_db, cell_id, year)
setkey(pred_all, cell_id, year)
pred_db[pred_all, consolidated := preds]

# Save results efficiently
fwrite(pred_db, "RF_imputated_db.csv")
```

---

### **Why This is Better**
- `data.table` filtering and joins avoid repeated memory copying (much faster than `dplyr` for large data).
- `mclapply` runs predictions in parallel across available cores.
- Predicts in chunks per year without loading unnecessary data into memory.
- Final write uses `fwrite`, which is extremely fast for large files.

This approach should make the computation feasible on a standard machine while preserving the pre-trained models.
END_RESPONSE>>>
