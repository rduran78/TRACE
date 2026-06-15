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
run_id: Anthropic_F01_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary heavy libraries loaded:** `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Unnecessary spatial data read:** `prep_data` is read from a shapefile but never used. Loading a large `.shp` file into memory is wasteful.

3. **Loading a potentially massive `.RData` file entirely into memory:** The `.RData` file likely contains training data (`joined_data`), the prediction database (`pred_db`), and the models (`rf_models_per_year`). The training data is no longer needed for prediction but remains in memory.

4. **Variable masking bug in the loop:** `filter(year == year)` is comparing the column `year` to itself (always `TRUE`) because the loop variable `year` is masked by the column name. This means `test_set` contains **all rows** every iteration, not just the rows for that year. This causes the prediction to run on the entire dataset for every year â€” dramatically increasing computation time and producing incorrect results.

5. **Using a full `data.frame` for prediction:** `randomForest::predict` on hundreds of thousands of rows in a single call can spike memory. No chunking is used.

6. **Writing a massive CSV at the end:** `write.csv` on a very large data frame is slow; `data.table::fwrite` is far faster.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Training data in memory | Remove `joined_data` (and any other unneeded objects) from the environment after loading `.RData`, then call `gc()` |
| Variable masking bug (`year == year`) | Rename the loop variable (e.g., `yr`) so the filter works correctly |
| Large single prediction call | Process predictions in chunks (e.g., 50,000 rows) to keep peak memory bounded |
| Slow `write.csv` | Use `data.table::fwrite` |
| General memory | Call `gc()` after removing large objects and between years |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(tidyverse)
library(data.table)

# â”€â”€ Load saved models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Free objects that are not needed for prediction â”€â”€
# (Adjust object names if your .RData contains different names)
if (exists("prep_data"))    rm(prep_data)
if (exists("joined_data"))  rm(joined_data)
# Remove any other large training-phase objects here, e.g.:
# if (exists("train_set")) rm(train_set)
gc()

# â”€â”€ Confirm required objects exist â”€â”€
stopifnot(exists("pred_db"), exists("rf_models_per_year"))

# â”€â”€ Initialise the output column â”€â”€
pred_db$consolidated <- NA_real_

# â”€â”€ Chunked prediction function (controls peak memory) â”€â”€
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, , drop = FALSE])
  }
  preds
}

# â”€â”€ Prediction loop (note: loop variable is 'yr' to avoid masking) â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, " ... "))

  year_key <- as.character(yr)


  if (!year_key %in% names(rf_models_per_year)) {
    cat("WARNING â€“ no model found, skipping.\n")
    next
  }

  # Subset only the rows for this year
  idx <- which(pred_db$year == yr)
  test_set <- pred_db[idx, , drop = FALSE]

  # Predict in chunks to limit memory spikes
  rf_model <- rf_models_per_year[[year_key]]
  pred_db$consolidated[idx] <- predict_chunked(rf_model, test_set)

  # Tidy up within the loop
  rm(test_set)
  gc()

  cat(paste0(length(idx), " cells done.\n"))
}

# â”€â”€ Write output (fwrite is ~5-10Ã— faster than write.csv) â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done. Output written to RF_imputated_db.csv\n")
```

---

### Summary of Changes

- **Removed** 10 unused library calls and the unused `st_read`.
- **Fixed the critical bug:** renamed loop variable to `yr` so `pred_db$year == yr` filters correctly instead of returning all rows every iteration.
- **Freed training-phase objects** immediately after loading, reclaiming potentially gigabytes of RAM.
- **Added chunked prediction** (`predict_chunked`) so that no single `predict()` call operates on more than 50,000 rows at a time, bounding peak memory.
- **Replaced `write.csv` with `fwrite`** for substantially faster I/O.
- **Added `gc()` calls** between years to return freed memory to the OS promptly.

These changes preserve all trained Random Forest models untouched and should make the workflow feasible on a standard personal computer.
END_RESPONSE>>>
