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
run_id: Anthropic_F01_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Loading a potentially massive `.RData` file entirely into memory:** The file `model_5_all_countries.RData` likely contains the training data (`joined_data`), the prediction frame (`pred_db`), and all year-specific Random Forest models (`rf_models_per_year`). All of these reside in memory simultaneously.

4. **Variable masking bug:** Inside the `for` loop, `filter(year == year)` compares the column `year` to itself (due to name collision with the loop variable), returning **all rows** instead of the intended subset. This means every year's prediction runs on the entire dataset, which is both incorrect and extremely slow/memory-intensive.

5. **Using `data.frame` operations on large data:** `pred_db$consolidated[pred_db$year == year] <- ...` triggers repeated full-column scans and copy-on-modify behavior in base R.

6. **Writing a potentially huge CSV in one shot:** For hundreds of thousands of rows Ã— many columns, `write.csv` is slow and memory-hungry.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so `dplyr::filter` correctly subsets |
| Entire dataset in memory at once | Process year-by-year, writing results incrementally, and free memory after each iteration |
| Slow base R assignment into `pred_db` | Convert to `data.table` and use keyed in-place update |
| Large CSV write | Use `data.table::fwrite` for fast, memory-efficient output |
| All RF models loaded simultaneously | Optionally, if `.RData` can be restructured, save/load one model at a time; otherwise, remove each model from the list after use to free memory |

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # required for predict.randomForest
library(dplyr)          # used only for the filter step; can be replaced with data.table

# â”€â”€ Do NOT load the unused shapefile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# prep_data <- st_read(...)
# Removed: saves potentially gigabytes of RAM.

# â”€â”€ Load saved objects â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# This brings in: rf_models_per_year, pred_db, joined_data (at minimum)
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert to data.table for fast keyed operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# Create a key on year for fast subsetting
setkey(pred_db,    year)
setkey(joined_data, year)

# â”€â”€ Year-by-year prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_to_predict <- sort(unique(pred_db$year))

for (yr in years_to_predict) {                       # 'yr' avoids name collision
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Subset using data.table key (very fast)
    test_set <- joined_data[.(yr)]                   # correct subset for this year

    # Predict
    preds <- predict(rf_model, newdata = test_set)

    # In-place update by reference (no copy)
    pred_db[.(yr), consolidated := preds]

    # Free the model from the list to release memory immediately
    rf_models_per_year[[yr_char]] <- NULL
    rm(rf_model, test_set, preds)
    gc()                                             # reclaim memory
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Fast, memory-efficient write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|--------|---------|
| 1 | Removed 11 unused libraries | Lower memory baseline |
| 2 | Removed unused `st_read` call | Avoids loading a large spatial object |
| 3 | Renamed loop variable to `yr` | **Fixes the bug** where `filter(year == year)` selected all rows |
| 4 | Converted to `data.table` with `setkey` | Subsetting and in-place update by reference; no copy-on-modify |
| 5 | Nullified each model after use (`rf_models_per_year[[yr_char]] <- NULL`) | Frees RAM progressively as predictions complete |
| 6 | Added `gc()` per iteration | Encourages R to return freed memory to the OS |
| 7 | Replaced `write.csv` with `fwrite` | Significantly faster and lower peak memory for large outputs |

No models are retrained; all existing Random Forest models are preserved and used as-is for prediction.
END_RESPONSE>>>
