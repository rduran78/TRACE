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
run_id: Anthropic_F01_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (rather than to the loop variable), returning *all* rows every iteration. This means every year's prediction runs on the entire dataset â€” massively inflating both memory use and compute time.

2. **Unnecessary spatial data load:** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes significant memory for no benefit.

3. **Full data frame kept in memory:** `joined_data` and `pred_db` may both be very large data frames with geometry columns and many predictor variables. Holding everything in memory at once is wasteful when predictions are done year-by-year.

4. **Data frame column assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a large data frame inside a loop triggers repeated full-column scans and copy-on-modify behavior in base R.

5. **Writing a single massive CSV at the end:** For hundreds of thousands of rows Ã— many columns, `write.csv()` is slow and memory-hungry.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) so dplyr filters correctly. |
| Unused `st_read()` call | Remove it entirely. |
| Entire dataset in memory | Convert to `data.table`; subset by year efficiently using keyed joins. Drop unneeded columns (especially geometry) before prediction. |
| Slow column assignment in loop | Use `data.table`'s set-by-reference (`:=`) with keyed subsetting. |
| Large CSV write | Use `data.table::fwrite()` for dramatically faster I/O. |
| Optional: memory pressure | Process and write results year-by-year (append mode) so peak memory is reduced. |

---

## Optimized Working R Code

```r
library(randomForest)
library(data.table)

# ------------------------------------------------------------------
# 1. Load only what is needed
# ------------------------------------------------------------------
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: pred_db, joined_data, rf_models_per_year

# 2. Convert to data.table for fast, memory-efficient operations
#    Drop any sf geometry if present â€” not needed for prediction
if (inherits(joined_data, "sf")) {
  joined_data <- as.data.table(sf::st_drop_geometry(joined_data))
} else {
  setDT(joined_data)
}

if (inherits(pred_db, "sf")) {
  pred_db <- as.data.table(sf::st_drop_geometry(pred_db))
} else {
  setDT(pred_db)
}

# 3. Key both tables on year for fast subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# 4. Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# 5. Identify which years actually have a trained model
available_years <- as.integer(names(rf_models_per_year))
loop_years      <- intersect(unique(pred_db$year), available_years)

# 6. Predict year-by-year using data.table keyed subsetting
for (yr in loop_years) {
  cat(sprintf("Predicting for year %d ...\n", yr))

  # Fast keyed subset â€” only the rows for this year
  test_set <- joined_data[.(yr)]

  rf_model <- rf_models_per_year[[as.character(yr)]]

  # Predict and assign by reference (no copy of the whole column)
  preds <- predict(rf_model, newdata = test_set)
  pred_db[.(yr), consolidated := preds]

  # Free memory immediately
  rm(test_set, preds)
  gc()
}

# 7. Warn about years with no model
missing <- setdiff(unique(pred_db$year), available_years)
if (length(missing) > 0) {
  cat("Warning: No model found for years:", paste(missing, collapse = ", "), "\n")
}

# 8. Fast write
data.table::fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

- **Bug fix:** The year-filtering shadowing issue is resolved by using a distinctly named loop variable (`yr`) and `data.table` keyed subsetting (`[.(yr)]`).
- **Memory savings:** Geometry columns are dropped; `data.table` operates by reference instead of copying; `gc()` is called each iteration.
- **Speed gains:** `data.table` keyed lookups are O(log n) instead of full scans; `fwrite()` is multithreaded and far faster than `write.csv()`.
- **Models untouched:** No Random Forest model is retrained or modified. Only `predict()` is called.
END_RESPONSE>>>
