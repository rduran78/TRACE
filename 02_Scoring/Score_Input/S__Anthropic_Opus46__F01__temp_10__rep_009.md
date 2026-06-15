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
run_id: Anthropic_F01_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a potentially massive `.RData` file entirely into memory:** The file `model_5_all_countries.RData` likely contains training data (`joined_data`), all year-specific Random Forest models (`rf_models_per_year`), and the prediction database (`pred_db`). All of these reside in memory simultaneously.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which means it evaluates to `TRUE` for every row (self-comparison) instead of filtering to the current loop iteration's year. This returns the entire dataset every iteration, causing the Random Forest `predict()` call to process all rows instead of one year's subset, drastically increasing computation time and producing incorrect results.

4. **Whole-dataframe column assignment in a loop:** Writing into `pred_db$consolidated[pred_db$year == year]` inside a loop over years is slow for large data frames due to repeated memory copying.

5. **`predict.randomForest` on hundreds of thousands of rows** is CPU-intensive and single-threaded by default; chunking or parallelism is not used.

6. **`st_read` for `prep_data`:** A large shapefile is loaded but never used in the prediction code, wasting memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused libraries and objects | Remove unused `library()` calls; remove `prep_data` load; after extracting needed objects from `.RData`, remove unneeded ones with `rm()` and `gc()` |
| Variable masking bug | Rename the loop variable (e.g., `yr`) so `filter(year == yr)` works correctly |
| Whole-frame writes in a loop | Collect predictions into a pre-allocated list, then bind once |
| Memory pressure from large `joined_data` | Convert `joined_data` to `data.table`; subset by year using keyed fast lookups; drop columns not in the model |
| Single-threaded `predict()` | Process years in parallel via `future.apply` or chunk large years |
| CSV output of huge frame | Use `data.table::fwrite()` for much faster writing |

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)
library(dplyr)            # only for select/filter if needed

# â”€â”€ 1. Load saved models and data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# This loads rf_models_per_year, joined_data, pred_db (and possibly others)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ 2. Identify predictor columns the models expect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Use the first available model to get the list of required predictors
sample_model  <- rf_models_per_year[[1]]
required_vars <- rownames(importance(sample_model))

# â”€â”€ 3. Trim joined_data to only required columns + year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
keep_cols <- intersect(c("year", required_vars), names(joined_data))
joined_dt <- as.data.table(joined_data)[, ..keep_cols]
setkey(joined_dt, year)

# Free the original large copy
rm(joined_data)
gc()

# â”€â”€ 4. Convert pred_db to data.table for fast indexed writes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_dt <- as.data.table(pred_db)
pred_dt[, consolidated := NA_real_]
setkey(pred_dt, year)

rm(pred_db)
gc()

# â”€â”€ 5. Predict year-by-year (correctly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- sort(unique(pred_dt$year))
model_years      <- names(rf_models_per_year)

for (yr in years_to_predict) {
  yr_char <- as.character(yr)
  cat(paste0("Predicting for year ", yr, " ... "))

  if (yr_char %in% model_years) {
    # Fast keyed subset â€“ only rows for this year
    test_set <- joined_dt[.(yr)]

    rf_model <- rf_models_per_year[[yr_char]]

    preds <- predict(rf_model, newdata = test_set)

    # Fast update by reference (no copy)
    pred_dt[.(yr), consolidated := preds]

    cat(paste0(nrow(test_set), " cells done.\n"))

    # Free per-iteration objects
    rm(test_set, preds)
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ 6. Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_dt, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Bug fix:** Renamed the loop iterator to `yr` so `joined_dt[.(yr)]` correctly filters to one year instead of returning the entire dataset.
- **Memory:** Removed unused library loads and the unused `prep_data` shapefile. Dropped unneeded columns from `joined_data`. Used `rm()` and `gc()` aggressively.
- **Speed:** Converted data frames to `data.table` with keys for O(log n) subsetting. Used `:=` assignment by reference (zero-copy update) instead of base R indexed assignment. Replaced `write.csv` with `fwrite`.
- **Models preserved:** No models are retrained; `rf_models_per_year` is loaded and used as-is.

These changes alone should reduce memory usage substantially and cut prediction time significantly, making the workflow feasible on a standard personal computer.
END_RESPONSE>>>
