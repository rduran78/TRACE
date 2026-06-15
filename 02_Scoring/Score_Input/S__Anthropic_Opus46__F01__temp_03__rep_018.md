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
run_id: Anthropic_F01_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes memory, especially if the shapefile is large.

2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on the full dataset â€” massively inflating computation time and producing incorrect results.

3. **Predicting on the entire dataset at once per year:** Even after fixing the bug, calling `predict()` on hundreds of thousands of rows with a large Random Forest model can spike memory usage because the model must route every row through every tree simultaneously.

4. **Using a base `data.frame` for column assignment in a loop:** Repeated assignment to `pred_db$consolidated[pred_db$year == year]` on a very large data.frame is slow due to repeated logical indexing and potential memory copying.

5. **Writing a massive CSV at the end:** `write.csv()` on a very large data.frame is slow; `data.table::fwrite()` is significantly faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused `st_read()` call | Remove it to free memory. |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) to avoid ambiguity. |
| Memory spike during `predict()` | Predict in chunks (batched prediction) to cap peak memory. |
| Slow indexing on `data.frame` | Convert `pred_db` to a `data.table` and use keyed joins or indexed updates. |
| Slow CSV write | Use `data.table::fwrite()`. |
| Unnecessary libraries loaded | Remove unused libraries to reduce overhead. |

The trained Random Forest models are **not retrained** â€” they are loaded and used as-is.

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict() on randomForest objects
library(tidyverse)      # only if joined_data is a tibble; otherwise removable

# â”€â”€ 1. Load the saved models and associated data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    (This loads rf_models_per_year, joined_data, pred_db, etc.)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ 2. Convert pred_db and joined_data to data.tables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ 3. Define a chunked predict function to limit memory usage â”€â”€â”€â”€â”€â”€â”€
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  
  # Pre-allocate result vector
  preds <- numeric(n)
  
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# â”€â”€ 4. Loop over years with correct subsetting and chunked prediction â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))
  
  yr_char <- as.character(yr)
  
  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]
    
    # Correct filter: use the loop variable, not the column name
    test_set <- joined_data[year == yr]
    
    # Chunked prediction to cap peak memory
    preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)
    
    # Fast indexed update via data.table
    pred_db[year == yr, consolidated := preds]
    
    # Free memory after each year
    rm(preds, test_set)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ 5. Write output with fast fwrite â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

- **Removed** the unused `st_read()` call and unnecessary library loads â†’ frees GBs of RAM.
- **Fixed the variable-shadowing bug** by renaming the loop variable to `yr` and using `data.table` syntax `joined_data[year == yr]`, which correctly references the loop variable.
- **Chunked prediction** (`predict_chunked`) processes rows in batches of 50,000 (tunable), preventing memory from spiking when the model scores hundreds of thousands of rows at once.
- **Converted to `data.table`** for fast subsetting and in-place column updates (no copy-on-modify).
- **Explicit `gc()`** after each year to return memory to the OS promptly.
- **`fwrite()`** replaces `write.csv()` for dramatically faster file output.

These changes make the workflow feasible on a normal machine without retraining any model.
END_RESPONSE>>>
