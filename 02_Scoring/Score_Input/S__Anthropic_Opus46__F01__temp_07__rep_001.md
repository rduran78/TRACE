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
run_id: Anthropic_F01_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning *all* rows every iteration. This means the full dataset is predicted repeatedly for every unique year â€” a massive waste of computation and likely the primary cause of slowness.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes significant memory for no benefit.

3. **Monolithic `predict()` calls on huge data frames:** With hundreds of thousands of rows and many predictors, a single `predict()` call on a Random Forest can spike memory because every tree must be evaluated across all rows simultaneously.

4. **Using `tidyverse` filtering inside a loop:** `dplyr::filter()` on a large data frame inside a tight loop adds overhead compared to index-based or `data.table` subsetting.

5. **Writing the full data frame to CSV at the end:** For very large data, `write.csv()` is slow; `data.table::fwrite()` is substantially faster.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so the filter actually subsets by year. |
| Unused shapefile load | Remove the `st_read()` call entirely. |
| Memory-heavy predict | Predict in row-chunks (e.g., 50 000 rows at a time) to cap peak memory. |
| Slow subsetting | Convert working data to `data.table` and use keyed subsetting. |
| Slow CSV write | Use `data.table::fwrite()`. |
| Library bloat | Load only the packages actually needed for prediction. |

---

## Optimized R Code

```r
# â”€â”€ Only load what is actually needed for prediction â”€â”€
library(data.table)
library(randomForest)

# â”€â”€ Load the pre-trained models â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert both frames to data.table for fast ops â”€â”€
pred_dt   <- as.data.table(pred_db)
joined_dt <- as.data.table(joined_data)

# â”€â”€ Pre-allocate the output column â”€â”€
pred_dt[, consolidated := NA_real_]

# â”€â”€ Set keys for fast subsetting â”€â”€
setkey(pred_dt,   year)
setkey(joined_dt, year)

# â”€â”€ Chunked prediction function (controls peak memory) â”€â”€
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# â”€â”€ Main prediction loop â”€â”€
years <- unique(pred_dt$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))
  
  yr_char <- as.character(yr)
  
  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€” skipping.\n", yr))
    next
  }
  
  # Correct subsetting: 'yr' is the loop variable, not the column name
  test_chunk <- joined_dt[year == yr]
  
  if (nrow(test_chunk) == 0L) {
    cat(sprintf("  Warning: No test data for year %s â€” skipping.\n", yr))
    next
  }
  
  rf_model <- rf_models_per_year[[yr_char]]
  
  preds <- predict_chunked(rf_model, test_chunk, chunk_size = 50000L)
  
  pred_dt[year == yr, consolidated := preds]
  
  cat(sprintf("  Done â€” %d predictions written.\n", length(preds)))
}

# â”€â”€ Fast write â”€â”€
fwrite(pred_dt, "RF_imputated_db.csv")

cat("All predictions complete. File written.\n")
```

---

## Summary of Changes

- **Bug fix:** The loop variable is now `yr`, so `joined_dt[year == yr]` correctly subsets to one year at a time instead of returning the entire dataset.
- **Removed unused `st_read()`** call â€” saves potentially gigabytes of RAM.
- **Chunked `predict()`** caps peak memory by processing 50 000 rows at a time (tunable via `chunk_size`).
- **`data.table`** keyed subsetting replaces `dplyr::filter()` for faster row selection.
- **`fwrite()`** replaces `write.csv()` for significantly faster output.
- **Reduced library loads** to only `data.table` and `randomForest`, lowering startup time and memory footprint.
- **No models are retrained** â€” only `predict()` is called on the existing `rf_models_per_year`.
END_RESPONSE>>>
