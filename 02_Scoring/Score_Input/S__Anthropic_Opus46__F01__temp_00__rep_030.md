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
run_id: Anthropic_F01_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Unnecessary spatial data load:** `prep_data` is read from a shapefile but never used in the prediction loop. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Loading a monolithic `.RData` file:** The `load()` call likely brings in not only `rf_models_per_year` and `pred_db`/`joined_data`, but also the full training data, intermediate objects, and possibly large spatial objects â€” all sitting in memory unnecessarily.

4. **Variable shadowing bug:** Inside `filter(year == year)`, the column `year` is compared to itself (the loop variable is masked by the column name), so `test_set` equals the **entire** `joined_data` every iteration, massively inflating prediction time and producing incorrect results.

5. **Predicting on a full `joined_data` data.frame:** If `joined_data` is an `sf` object or contains columns not needed by the model, unnecessary data is carried through each prediction call.

6. **Writing a potentially huge CSV at the end:** For hundreds of thousands of rows Ã— many columns, `write.csv` is slow and the output file is large.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce load time and memory |
| Unused `prep_data` shapefile | Remove the `st_read` call entirely |
| Monolithic `.RData` loads everything | Save only the needed objects (`rf_models_per_year`, `pred_db`, `joined_data`) to separate `.rds` files, or selectively load from the `.RData` using a temporary environment |
| Variable shadowing bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly, drastically reducing per-iteration data size |
| Carrying unneeded columns into `predict()` | Subset `joined_data` to only the predictor columns the model expects |
| Slow `write.csv` | Use `data.table::fwrite` for much faster I/O |
| Optional: memory pressure from all models loaded at once | Load/unload models one at a time if saved individually |

---

## Optimized R Code

```r
# â”€â”€ Only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(data.table)
library(dplyr)

# â”€â”€ 1. Load only the objects you need â”€â”€
# Use a temporary environment so stray objects in the .RData don't pollute RAM.
tmp_env <- new.env()
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData', envir = tmp_env)

rf_models_per_year <- tmp_env$rf_models_per_year
pred_db            <- tmp_env$pred_db
joined_data        <- tmp_env$joined_data

rm(tmp_env)            # free everything else that was in the .RData
gc()

# â”€â”€ 2. Remove the unused shapefile read entirely â”€â”€
# (prep_data was never used in prediction)

# â”€â”€ 3. Identify the predictor columns the RF models expect â”€â”€
# Pull the variable names from any one of the trained models.
example_model   <- rf_models_per_year[[1]]
predictor_names <- rownames(importance(example_model))

# Keep only what predict() needs â€” drops geometry, IDs, etc.
joined_data <- as.data.frame(joined_data)          # drop sf geometry if present
joined_data <- joined_data[, c("year", predictor_names), drop = FALSE]
gc()

# â”€â”€ 4. Pre-allocate the output column â”€â”€
pred_db$consolidated <- NA_real_

# â”€â”€ 5. Prediction loop with the variable-shadowing bug fixed â”€â”€
years_to_predict <- unique(pred_db$year)

for (yr in years_to_predict) {
  cat("Predicting for year", yr, "\n")

  # Correct filter: compare column 'year' to the loop variable 'yr'
  test_set <- joined_data[joined_data$year == yr, , drop = FALSE]

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    rf_model <- rf_models_per_year[[yr_char]]
    pred_db$consolidated[pred_db$year == yr] <- predict(rf_model, newdata = test_set)
  } else {
    cat("  Warning: No model found for year", yr, "\n")
  }
}

# â”€â”€ 6. Fast write with fwrite â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Expected Gains

| Change | Impact |
|---|---|
| Fix `year == year` filter bug | Reduces per-iteration rows from **N** (all years) to **N/T** (one year). This alone can yield a **5â€“20Ã— speedup** depending on the number of years. |
| Drop unused shapefile load | Saves potentially **gigabytes** of RAM. |
| Isolate `.RData` load in temp env | Frees all training-phase objects from memory. |
| Subset to predictor columns only | Reduces working data frame size substantially. |
| Remove unused libraries | Faster startup, lower baseline memory. |
| `fwrite` instead of `write.csv` | **5â€“10Ã— faster** file writing. |

No models are retrained; all original `rf_models_per_year` objects are preserved and used as-is.
END_RESPONSE>>>
