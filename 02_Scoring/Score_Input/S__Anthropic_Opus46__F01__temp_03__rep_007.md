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
run_id: Anthropic_F01_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction loop. This wastes significant memory, especially for hundreds of thousands of geographic cells.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` contains **all rows** every iteration â€” massively inflating computation and producing incorrect predictions.

4. **Using `predict()` on the full `joined_data` tibble:** Random Forest `predict()` on a large `tibble`/`data.frame` with many unused columns is slower and more memory-intensive than necessary. Only the predictor columns should be passed.

5. **Writing a potentially huge CSV with `write.csv`:** For very large data, `data.table::fwrite` is far faster and more memory-efficient.

6. **No garbage collection or memory management:** The `.RData` file likely contains the training data (`joined_data`), all models, and other objects. These all persist in memory simultaneously.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused library calls | Reduce memory footprint |
| Remove unused `prep_data` shapefile read | Free potentially gigabytes of RAM |
| Fix the `year == year` variable masking bug | Correctness â€” also prevents predicting on the entire dataset each iteration |
| Extract only required predictor columns for `predict()` | Reduce memory passed to the model |
| Use `data.table::fwrite` instead of `write.csv` | Much faster I/O for large files |
| Call `gc()` after removing unneeded objects | Reclaim memory |
| Process predictions in chunks if needed | Prevent memory spikes |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(data.table)
library(dplyr)

# â”€â”€ Load the saved models and associated data â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load the shapefile â€” it is unused in prediction.
# prep_data <- st_read(...)  # REMOVED

# â”€â”€ Identify the predictor variable names from one of the trained models â”€â”€
sample_model <- rf_models_per_year[[names(rf_models_per_year)[1]]]
predictor_vars <- rownames(importance(sample_model))

# â”€â”€ Remove any large objects from the .RData that are not needed â”€â”€
# Keep only: pred_db, joined_data, rf_models_per_year
# (Adjust object names if the .RData contains other large objects)
objects_to_keep <- c("pred_db", "joined_data", "rf_models_per_year",
                     "predictor_vars", "sample_model")
rm(list = setdiff(ls(), objects_to_keep))
rm(sample_model)
gc()

# â”€â”€ Convert joined_data to data.table for faster subsetting â”€â”€
setDT(joined_data)
setDT(pred_db)

# â”€â”€ Initialise the output column â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Prediction loop with the variable-masking bug fixed â”€â”€
years_to_predict <- unique(pred_db$year)

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]

    # Subset only the rows for this year and only the predictor columns
    test_set <- joined_data[year == yr, ..predictor_vars]

    # Predict
    preds <- predict(rf_model, newdata = test_set)

    # Assign predictions back
    pred_db[year == yr, consolidated := preds]

    # Free memory from this iteration
    rm(test_set, preds)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Write output efficiently â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Changes

| Original Issue | Fix Applied |
|---|---|
| `filter(year == year)` selects all rows (masking bug) | Renamed loop variable to `yr`; used `data.table` syntax `joined_data[year == yr, ...]` |
| Full `joined_data` passed to `predict()` | Only predictor columns (`..predictor_vars`) are passed |
| Unused shapefile loaded into memory | Removed entirely |
| 14 libraries loaded; most unused | Reduced to 3 (`randomForest`, `data.table`, `dplyr`) |
| `write.csv` on large data | Replaced with `data.table::fwrite` |
| No memory management | Added `rm()` and `gc()` calls after each year and after setup |

These changes preserve all trained Random Forest models exactly as-is and require no retraining.
END_RESPONSE>>>
