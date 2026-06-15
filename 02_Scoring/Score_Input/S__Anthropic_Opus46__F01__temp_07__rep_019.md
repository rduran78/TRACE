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
run_id: Anthropic_F01_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used, consuming memory for nothing.
2. **Variable shadowing bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (the column name shadows the loop variable), so `test_set` will always equal the entire `joined_data` dataframe rather than a single year's subset. This means every iteration predicts on *all* rows â€” massively inflating computation time.
3. **Entire `joined_data` held in memory:** With hundreds of thousands of cells Ã— many years Ã— many predictors, this is a large object. Prediction on all of it (due to the bug) makes it worse.
4. **`predict.randomForest` on huge data:** Random Forest prediction scales with `n_rows Ã— n_trees Ã— n_variables`. Doing this repeatedly (and on the wrong-sized subset) is very expensive.
5. **Writing a single monolithic CSV at the end:** For a very large `pred_db`, `write.csv` is slow. `data.table::fwrite` is significantly faster.
6. **No garbage collection or memory management:** All models for all years are loaded simultaneously via `rf_models_per_year`, which may be very large.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Unused `st_read` call | Remove it |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Predicting on full data every iteration | Fix the filter so only one year's rows are predicted |
| Memory pressure from all RF models loaded at once | Optionally load/unload models one at a time if saved separately; at minimum, `gc()` after each year |
| Slow CSV write | Use `data.table::fwrite` |
| `joined_data` as tibble | Convert to `data.table` for faster subsetting |
| No progress feedback on row counts | Add diagnostic `cat()` messages |

---

## Optimized R Code

```r
# â”€â”€ Only load what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load the saved workspace (models + data) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Contains: rf_models_per_year (list of trained RF models),
#           joined_data (predictor data), pred_db (output scaffold)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Remove any large objects from that workspace that are not needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Uncomment and adapt if other large objects were saved in the .RData file:
# rm(object_not_needed); gc()

# â”€â”€ Convert to data.table for fast keyed subsetting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(joined_data)
setDT(pred_db)

# Create a key on year for O(log n) subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# â”€â”€ Prediction loop (year by year) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(paste0("Warning: No model found for year ", yr, " â€” skipping.\n"))
    next
  }

  cat(paste0("Predicting for year ", yr, " ... "))

  # ---- Correct, fast subset (no variable-shadowing bug) ----
  test_set <- joined_data[.(yr)]            # keyed lookup, very fast

  rf_model <- rf_models_per_year[[yr_char]]

  preds <- predict(rf_model, newdata = test_set)

  pred_db[.(yr), consolidated := preds]

  cat(paste0(length(preds), " cells done.\n"))

  # ---- Free memory eagerly ----
  rm(test_set, preds)
  gc()
}

# â”€â”€ Optionally free the models list if no longer needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(rf_models_per_year, joined_data)
gc()

# â”€â”€ Fast write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Impact |
|---|---|---|
| 1 | Removed unused `st_read()` call | Saves memory (large shapefile never loaded) |
| 2 | Renamed loop variable to `yr`; used `joined_data[.(yr)]` | **Fixes the shadowing bug** â€” the single biggest performance problem |
| 3 | Converted to `data.table` with `setkey` | Subsetting goes from O(n) scan to O(log n) keyed lookup |
| 4 | Added `rm()` + `gc()` inside the loop | Frees prediction vectors each iteration |
| 5 | Replaced `write.csv` with `fwrite` | 5â€“10Ã— faster file writing |
| 6 | Removed 12 unused `library()` calls | Faster startup, smaller memory footprint |
| 7 | No models retrained | All original RF models preserved and reused as-is |

These changes should make the script feasible on a standard personal computer without any modification to the trained Random Forest models.
END_RESPONSE>>>
