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
run_id: OpenAI_F01_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F01_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
- The inefficiency mainly comes from:  
  1. Looping through each year sequentially and predicting in-memory with very large `test_set`.  
  2. Using `dplyr` filtering repeatedly on a huge dataset (`joined_data`), which creates many large intermediate objects.  
  3. Operating fully in Râ€™s main memory with large shapefile and prediction frames.  

**Optimization Strategy:**  
- Use `data.table` for subsetting and updates (very memory-efficient).  
- Avoid copying the data repeatedly â€” process by reference.  
- Predict in chunks instead of all at once to prevent memory overflow.  
- Use `fread`/`fwrite` for fast I/O instead of `write.csv`.  
- Multi-threading is not possible with standard `randomForest::predict`, but chunking helps avoid crashes.  

**Revised Efficient R Code:**  
```r
library(data.table)
library(randomForest)

# Read data efficiently
prep_data <- st_read('/Volumes/Toshi 1Tb/Amaz/geographic_cell_data/geographic_cell_data.shp')

# Load trained models
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Convert pred_db and joined_data to data.table
setDT(pred_db)
setDT(joined_data)

# Ensure consolidated column exists
pred_db[, consolidated := NA_real_]

# Function for chunked prediction
predict_in_chunks <- function(model, data, chunk_size = 50000) {
  n <- nrow(data)
  res <- numeric(n)
  idx_seq <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (idx in idx_seq) {
    res[idx] <- predict(model, newdata = data[idx])
  }
  res
}

# Predict for each year
years <- unique(pred_db$year)

for (yr in years) {
  cat("Predicting for year", yr, "\n")
  
  if (as.character(yr) %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[as.character(yr)]]
    
    # Subset by reference
    idx <- which(pred_db$year == yr)
    test_set <- joined_data[year == yr]
    
    # Chunked prediction
    preds <- predict_in_chunks(rf_model, test_set)
    
    # Assign predictions efficiently
    pred_db[idx, consolidated := preds]
  } else {
    cat("Warning: No model found for year", yr, "\n")
  }
}

# Fast write
fwrite(pred_db, "RF_imputated_db.csv")
```

**Key Gains:**  
- Eliminates expensive `dplyr` operations â†’ replaced with `data.table` (in-place updates).  
- Implements chunked prediction to avoid memory overload.  
- Uses `fwrite` for fast output instead of `write.csv`.  
- Maintains trained models and prediction integrity.  

This approach will significantly reduce both **memory footprint** and **runtime** on a standard personal computer.
END_RESPONSE>>>
