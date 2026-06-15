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
run_id: Anthropic_F01_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F01_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read from a shapefile but never used in the prediction loop. This wastes significant memory, especially if the shapefile is large.

3. **Variable masking bug:** `filter(year == year)` inside the loop does not work as intended. The loop variable `year` shadows the column name `year`, so the filter matches *all* rows every iteration rather than just the target year. This means the `predict()` call processes the entire dataset on every iteration, massively inflating computation time.

4. **Predicting on the full dataset repeatedly:** Because of the bug above, every year's prediction pass sends hundreds of thousands (or millions) of unnecessary rows through the Random Forest model.

5. **Using a `data.frame` for row-level assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` on a very large data frame inside a loop is slow due to repeated indexing and copy-on-modify semantics.

6. **`randomForest::predict` is single-threaded and memory-heavy on large data:** Sending all rows at once (or too many rows due to the bug) can spike memory.

---

## Optimization Strategy

| Action | Rationale |
|---|---|
| Remove unused libraries and the unused `prep_data` object | Free memory |
| Fix the variable masking bug by renaming the loop variable | Ensure only the correct year's rows are predicted |
| Convert to `data.table` for fast subsetting and assignment | Avoid copy-on-modify overhead |
| Optionally batch large year-groups into chunks | Cap peak memory during `predict()` |
| Use `fwrite` instead of `write.csv` | Much faster I/O for large files |
| Call `gc()` between years | Release memory promptly |

The trained Random Forest models are **not retrained** â€” only the prediction loop is optimized.

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data is a tibble / uses dplyr structures

# â”€â”€ Load the pre-trained models and associated data â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load the shapefile â€” it is unused in prediction.
# prep_data <- st_read(...)   # REMOVED to save memory

# â”€â”€ Convert to data.table for fast indexed operations â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ Prediction loop (bug-fixed and optimized) â”€â”€
# Rename loop variable to avoid masking the column name "year"
years_to_predict <- unique(pred_db$year)

# Optional: set a chunk size to cap memory during predict()
CHUNK_SIZE <- 50000L    # adjust based on available RAM

for (yr in years_to_predict) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!(yr_char %in% names(rf_models_per_year))) {
    cat(paste0("  Warning: No model found for year ", yr, " â€” skipping.\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correct filter: use the renamed loop variable 'yr'
  test_set <- joined_data[year == yr]

  n <- nrow(test_set)

  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, " â€” skipping.\n"))
    next
  }

  # Predict in chunks to limit peak memory
  if (n <= CHUNK_SIZE) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    preds <- numeric(n)
    starts <- seq(1L, n, by = CHUNK_SIZE)
    for (i in seq_along(starts)) {
      idx_start <- starts[i]
      idx_end   <- min(idx_start + CHUNK_SIZE - 1L, n)
      preds[idx_start:idx_end] <- predict(rf_model,
                                           newdata = test_set[idx_start:idx_end])
    }
  }

  # Fast indexed assignment via data.table
  pred_db[year == yr, consolidated := preds]

  # Free memory before next iteration
  rm(test_set, preds)
  gc()
}

# â”€â”€ Write output efficiently â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

| # | Change | Effect |
|---|--------|--------|
| 1 | Removed 10+ unused library calls and the unused shapefile read | Reduces memory footprint by potentially gigabytes |
| 2 | Renamed loop variable from `year` to `yr` | **Fixes the critical filter bug** â€” previously every year predicted on the *entire* dataset |
| 3 | Converted `pred_db` and `joined_data` to `data.table` | Faster subsetting (`[year == yr]`) and in-place column assignment (no deep copies) |
| 4 | Added chunked prediction (`CHUNK_SIZE`) | Caps peak memory during `predict()` for years with hundreds of thousands of rows |
| 5 | Replaced `write.csv` with `fwrite` | Orders-of-magnitude faster file writing |
| 6 | Added `gc()` after each year | Promptly reclaims memory |
| 7 | Models are **untouched** â€” no retraining | Preserves the original `rf_models_per_year` exactly as loaded |

These changes together should make the prediction process feasible on a standard personal computer with moderate RAM (8â€“16 GB).
END_RESPONSE>>>
