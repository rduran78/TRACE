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
run_id: Anthropic_F01_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (not to the loop variable), returning *all rows* every iteration. This means every year's prediction runs on the entire dataset â€” massively inflating compute time and producing incorrect results.

2. **Unnecessary Spatial Data Load:** `prep_data` is read via `st_read()` but never used, consuming significant memory for a large shapefile.

3. **Full Dataset in Memory:** `joined_data` and `pred_db` are likely large `data.frame` or `sf` objects. Holding them entirely in memory alongside the Random Forest models (which can be very large) may exceed RAM.

4. **Inefficient Row Assignment:** `pred_db$consolidated[pred_db$year == year]` performs a full-column logical scan on every iteration.

5. **Unused Libraries:** Many loaded libraries (`blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, `terra`, etc.) are never used in the prediction loop, adding overhead.

6. **CSV Output of Huge Data:** `write.csv()` on hundreds of thousands of rows is slow; `data.table::fwrite()` is far faster.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so `filter(year == yr)` works correctly |
| Unused shapefile load | Remove `st_read()` call |
| Unused libraries | Load only what is needed |
| Large objects in memory | Convert to `data.table`; drop unneeded columns; process year-by-year and free memory |
| Slow row indexing | Use `data.table` keyed joins / set-by-reference |
| Slow CSV write | Use `fwrite()` |
| Optional: parallelism | Not needed once the bug is fixed (each year's subset is now much smaller) |

The trained Random Forest models are **preserved untouched** â€” no retraining occurs.

---

## Optimized R Code

```r
# â”€â”€ Load only required libraries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict() on rf objects
library(dplyr)          # only if joined_data is a tibble; can be dropped if converted

# â”€â”€ Load pre-trained models and prediction database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (Do NOT load the unused shapefile)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: pred_db, joined_data, rf_models_per_year

# â”€â”€ Convert to data.table for speed and memory efficiency â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Pre-allocate the output column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Identify the predictor columns the models expect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (Grab from any one model so we only keep needed columns in the subset)
sample_model <- rf_models_per_year[[1]]
model_vars   <- rownames(importance(sample_model))

# â”€â”€ Key joined_data by year for fast subsetting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(joined_data, year)
setkey(pred_db, year)

# â”€â”€ Predict year-by-year â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {
  yr_char <- as.character(yr)
  cat(paste0("Predicting for year ", yr, " ... "))

  if (!yr_char %in% names(rf_models_per_year)) {
    cat("WARNING: no model found â€” skipping.\n")
    next
  }

  # Fast keyed subset; keep only columns the model needs (saves memory)
  test_set <- joined_data[.(yr), .SD, .SDcols = intersect(model_vars, names(joined_data))]

  rf_model <- rf_models_per_year[[yr_char]]
  preds    <- predict(rf_model, newdata = test_set)

  # Set by reference â€” no copy of the whole column
  pred_db[.(yr), consolidated := preds]

  cat(paste0(length(preds), " cells predicted.\n"))

  # Free temporary objects to keep memory low
  rm(test_set, preds)
  gc()
}

# â”€â”€ Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Impact |
|---|--------|--------|
| 1 | **Fixed the `year == year` bug** by using loop variable `yr` | Correctness + massive speed gain (subset is now 1/N-th the size) |
| 2 | Removed unused `st_read()` | Saves potentially GBs of RAM |
| 3 | Removed 11 unused `library()` calls | Faster startup, less memory |
| 4 | Converted to `data.table` with keys | O(log n) subsetting; set-by-reference avoids copies |
| 5 | Subsetted only model-required columns (`model_vars`) | Reduces per-iteration memory |
| 6 | Added `rm()` + `gc()` inside loop | Frees memory each iteration |
| 7 | Replaced `write.csv()` with `fwrite()` | ~5â€“10Ã— faster file writing |
| 8 | **No model retraining** | Models used as-is from the `.RData` file |

These changes should make the prediction loop feasible on a standard personal computer with moderate RAM (8â€“16 GB).
END_RESPONSE>>>
