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
run_id: Anthropic_F01_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable name shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (because the loop variable `year` shadows the column name), returning *all* rows instead of the intended subset. This means every iteration predicts on the entire dataset â€” massively wasteful and incorrect.

2. **Loading unnecessary spatial data:** `prep_data` is read via `st_read()` but never used, consuming memory for nothing.

3. **Entire `joined_data` held in memory as a data.frame/tibble:** With hundreds of thousands of rows and many predictors, `dplyr::filter()` on a large tibble each iteration is slower than necessary.

4. **Column assignment via logical indexing on a large data.frame (`pred_db$consolidated[pred_db$year == year]`):** This is slow for large data.frames because R copies the entire column on each assignment.

5. **`predict.randomForest` on a huge test set in one call:** For very large test sets, `predict()` can spike memory because the Random Forest must route every observation through every tree simultaneously.

6. **Many heavy libraries loaded but unused:** `blockCV`, `zoo`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra` â€” none are needed for prediction, yet they consume memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries consume memory | Remove all unnecessary `library()` calls |
| Unused `prep_data` spatial object | Remove the `st_read()` call entirely |
| Variable shadowing bug (`year == year`) | Rename the loop variable (e.g., `yr`) so filtering works correctly |
| Slow subsetting and assignment on data.frame | Convert `pred_db` and `joined_data` to `data.table` for fast keyed subsetting and update-by-reference |
| Potential memory spike in `predict()` | Predict in chunks (batches) to cap peak memory |
| Writing a huge CSV at the end | Use `data.table::fwrite()` for much faster I/O |

The trained Random Forest models (`rf_models_per_year`) are **preserved exactly as-is** â€” no retraining occurs.

---

## Optimized R Code

```r
# â”€â”€ Only the libraries actually needed for prediction â”€â”€
library(data.table)
library(randomForest)

# â”€â”€ Load the saved models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects: rf_models_per_year, pred_db, joined_data

# â”€â”€ Convert to data.table for speed and memory-efficient operations â”€â”€
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Initialise the output column â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Set keys for fast subsetting â”€â”€
setkey(pred_db, year)
setkey(joined_data, year)

# â”€â”€ Batch-size for chunked prediction (tune to your RAM) â”€â”€
BATCH_SIZE <- 50000L

# â”€â”€ Predict year by year â”€â”€
unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(sprintf("Predicting for year %s\n", yr))

  yr_char <- as.character(yr)


  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€” skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Fast keyed subset (no copy of the whole table)
  test_set <- joined_data[.(yr)]
  n <- nrow(test_set)

  if (n == 0L) {
    cat(sprintf("  No rows in joined_data for year %s â€” skipping.\n", yr))
    next
  }

  # â”€â”€ Chunked prediction to cap peak memory â”€â”€
  preds <- vector("numeric", n)
  starts <- seq(1L, n, by = BATCH_SIZE)

  for (s in starts) {
    e <- min(s + BATCH_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # â”€â”€ Update by reference (no column copy) â”€â”€
  pred_db[.(yr), consolidated := preds]

  # â”€â”€ Free memory immediately â”€â”€
  rm(test_set, preds, rf_model)
  gc()
}

# â”€â”€ Fast write â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Why |
|---|---|---|
| 1 | Removed 11 unused `library()` calls and the unused `st_read()` | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` | Fixes the `filter(year == year)` shadowing bug that caused incorrect (and bloated) subsets |
| 3 | Converted to `data.table` with keys | Subsetting and assignment by reference are orders of magnitude faster than base/dplyr on large frames |
| 4 | Chunked `predict()` calls (`BATCH_SIZE`) | Prevents a single massive allocation inside `predict.randomForest`; tune the batch size to your available RAM |
| 5 | Explicit `rm()` + `gc()` after each year | Releases memory from the previous year's test set before the next iteration |
| 6 | `fwrite()` instead of `write.csv()` | Typically 5â€“10Ã— faster for large files |

No models are retrained; `rf_models_per_year` is loaded and used read-only.
END_RESPONSE>>>
