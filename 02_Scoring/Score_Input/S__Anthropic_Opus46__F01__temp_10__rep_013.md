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
run_id: Anthropic_F01_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Cell-Level GDP Prediction R Script

---

## Diagnosis

There are several performance and correctness issues in the provided code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (because the loop variable and column name are identical), returning **all rows** instead of the intended subset. This means every year's prediction runs on the entire dataset, massively inflating computation.

2. **Loading Unnecessary Spatial Data:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes memory for no reason.

3. **Excessive Libraries:** Many loaded libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, etc.) are not used in the prediction step, consuming memory overhead.

4. **Full Data Frame in Memory:** `joined_data` and `pred_db` may be very large data frames held entirely in memory. Subsetting with `dplyr::filter` on a non-indexed data frame is slow for hundreds of thousands of rows repeated across many years.

5. **Row-by-Row Assignment Pattern:** Assigning predictions back into `pred_db` via logical indexing (`pred_db$consolidated[pred_db$year == year]`) rescans the entire data frame each iteration.

6. **CSV Output for Large Data:** `write.csv` on a very large data frame is slow and produces large files; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename loop variable to `yr` |
| Unused spatial data loaded | Remove `st_read()` call |
| Unnecessary libraries | Load only what is needed |
| Slow subsetting & assignment | Convert to `data.table`, key by `year`, and use keyed joins |
| Slow CSV write | Use `data.table::fwrite()` |
| Potential memory pressure | Process and predict one year at a time; `gc()` between iterations |
| Large predict overhead | Keep only predictor columns needed by the model for each `predict()` call |

---

## Optimized Working R Code

```r
# â”€â”€ Load only required libraries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load pre-trained models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert prediction database to data.table and key it â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# Initialise the output column
pred_db[, consolidated := NA_real_]

# Key both tables by year for fast subsetting
setkey(pred_db, year)
setkey(joined_data, year)

# â”€â”€ Identify predictor columns once (from any stored model) â”€â”€â”€â”€â”€â”€
sample_model   <- rf_models_per_year[[1]]
predictor_cols <- rownames(sample_model$importance)

# â”€â”€ Predict year by year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(pred_db$year))

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  # Fast keyed subset â€“ only the columns the model needs
  test_set <- joined_data[.(yr), ..predictor_cols]

  # Predict
  preds <- predict(rf_models_per_year[[yr_char]], newdata = test_set)

  # Fast keyed assignment
  pred_db[.(yr), consolidated := preds]

  # Free memory from this iteration
  rm(test_set, preds)
  gc()
}

# â”€â”€ Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

1. **Bug fix:** Renamed the loop variable from `year` to `yr` so the filter/subset correctly selects only the rows for that year.
2. **Removed unused code:** Eliminated the `st_read` call and all libraries not required for prediction.
3. **Switched to `data.table`:** Keyed subsetting (`[.(yr)]`) and column restriction (`..predictor_cols`) are much faster and more memory-efficient than `dplyr::filter` on large frames.
4. **Column pruning:** Only the predictor columns required by the Random Forest model are passed to `predict()`, reducing memory copying.
5. **Explicit garbage collection** (`gc()`) after each year frees temporary allocations.
6. **`fwrite` instead of `write.csv`:** Typically 5â€“10Ã— faster for large outputs.

These changes should make the workflow feasible on a standard personal computer without retraining any models.
END_RESPONSE>>>
