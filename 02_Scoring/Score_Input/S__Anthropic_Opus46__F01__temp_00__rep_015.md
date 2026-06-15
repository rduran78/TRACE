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
run_id: Anthropic_F01_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` equals the **entire** `joined_data` every iteration. This is both a correctness bug and a massive performance problem â€” every year predicts on the full dataset.

4. **Monolithic `predict()` on hundreds of thousands of rows:** Even after fixing the filter bug, calling `predict()` on a very large data frame in one shot can spike memory, especially with large Random Forest models (which store many trees).

5. **Using `data.frame` operations:** `pred_db` is likely a large data frame; indexed row assignment (`pred_db$consolidated[pred_db$year == year] <- ...`) is slow on large data frames.

6. **Writing a massive CSV at the end:** `write.csv()` is slow for large files.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read()` call entirely |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable (e.g., `yr`) |
| Memory spike during `predict()` | Predict in chunks (batches) to cap peak memory |
| Slow data.frame indexing | Convert `pred_db` to `data.table` for fast indexed assignment |
| Slow CSV writing | Use `data.table::fwrite()` instead of `write.csv()` |
| Optional: parallelism | Not pursued here to keep memory low on a personal machine |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(data.table)
library(dplyr)          # only for the initial filter; could also use data.table

# â”€â”€ Load the pre-trained models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# This should bring into scope: rf_models_per_year, pred_db, joined_data

# â”€â”€ Convert to data.table for fast indexed operations â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ Set a chunk size that fits comfortably in RAM â”€â”€
CHUNK_SIZE <- 50000L   # adjust based on available memory

# â”€â”€ Prediction loop â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  # Correct filter: compare column to the loop variable, not to itself
  test_set <- joined_data[year == yr]

  model_key <- as.character(yr)

  if (!model_key %in% names(rf_models_per_year)) {
    cat(paste0("  Warning: No model found for year ", yr, " â€” skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[model_key]]
  n <- nrow(test_set)

  # â”€â”€ Predict in chunks to limit peak memory â”€â”€
  preds <- numeric(n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    preds[s:e] <- predict(rf_model, newdata = test_set[s:e, ])
  }

  # â”€â”€ Fast indexed assignment via data.table â”€â”€
  pred_db[year == yr, consolidated := preds]

  # â”€â”€ Free memory after each year â”€â”€
  rm(test_set, preds)
  gc()
}

# â”€â”€ Optionally free the models if no longer needed â”€â”€
rm(rf_models_per_year, joined_data)
gc()

# â”€â”€ Fast write â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

| # | Change | Benefit |
|---|---|---|
| 1 | Removed 11 unused library calls and the unused `st_read()` | Frees hundreds of MB of RAM |
| 2 | Renamed loop variable to `yr` to fix the `filter(year == year)` self-comparison bug | **Correctness fix** â€” previously predicted on the entire dataset every iteration |
| 3 | Switched `pred_db` and `joined_data` to `data.table` | Much faster subsetting and assignment |
| 4 | Predict in configurable chunks (`CHUNK_SIZE`) | Caps peak memory; avoids out-of-memory crashes |
| 5 | Explicit `rm()` + `gc()` after each year | Returns memory to the OS promptly |
| 6 | Replaced `write.csv()` with `fwrite()` | Typically 5â€“10Ã— faster for large files |

No models are retrained; only the prediction and I/O pipeline is optimized.
END_RESPONSE>>>
