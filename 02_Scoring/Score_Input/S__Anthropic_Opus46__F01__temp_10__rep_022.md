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
run_id: Anthropic_F01_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Loading the entire spatial dataset unnecessarily.** `prep_data` is read via `st_read()` but never used in the prediction loop. Loading a large shapefile consumes significant memory for no benefit.

2. **Variable masking bug in `filter()`.** The line `filter(year == year)` compares the column `year` to itself (due to name collision with the loop variable), so it returns **all rows** instead of the subset for a single year. This means every iteration predicts on the entire dataset, massively increasing computation and producing incorrect results.

3. **Subsetting with `dplyr::filter()` on a potentially huge data frame each iteration.** Even once the bug is fixed, `filter()` on a tibble/data.frame for hundreds of thousands of rows inside a loop is slower than necessary.

4. **Row-by-row assignment into a large data frame with logical indexing (`pred_db$consolidated[pred_db$year == year]`) each iteration.** This triggers repeated full-column scans.

5. **`predict.randomForest` on a very large `newdata` can be memory-intensive**, especially if the forest has many trees and the data has many predictors. No chunking is performed.

6. **Keeping all loaded `.RData` objects in memory simultaneously** (e.g., `joined_data`, `pred_db`, `rf_models_per_year`, and possibly others) can exhaust RAM.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused `prep_data` spatial read | Remove it entirely |
| Variable masking bug (`year == year`) | Rename loop variable or use `.env$` pronoun |
| Slow subsetting & assignment in a `data.frame` | Convert `joined_data` and `pred_db` to `data.table` and use keyed subsetting |
| Large single-pass `predict()` call | Chunk prediction into batches (e.g., 50 000 rows) to cap peak memory |
| All objects in memory at once | Remove unneeded objects and call `gc()` after each year |
| Output as CSV for a huge table | Use `data.table::fwrite()` (much faster than `write.csv()`) |

**Key constraint honoured:** the trained Random Forest models (`rf_models_per_year`) are never retrained â€” only `predict()` is called.

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(data.table)
library(randomForest)   # needed for predict() on randomForest objects

# â”€â”€ 1. Load saved workspace (models + data) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove prep_data if it was loaded (unused); free memory immediately
if (exists("prep_data")) { rm(prep_data); gc() }

# â”€â”€ 2. Convert to data.table for fast keyed operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(joined_data)
setDT(pred_db)

# Create keyed index on year for O(log n) subsetting
setkey(joined_data, year)
setkey(pred_db, year)

# Pre-allocate the output column
pred_db[, consolidated := NA_real_]

# â”€â”€ 3. Chunked prediction function (caps peak memory) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
predict_chunked <- function(model, newdata, chunk_size = 50000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }
  # Pre-allocate result vector
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# â”€â”€ 4. Prediction loop (one year at a time) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€“ skipping.\n", yr))
    next
  }

  # Keyed subset â€“ fast, no copying of the full table
  test_set <- joined_data[.(yr)]

  rf_model <- rf_models_per_year[[yr_char]]

  # Chunked predict to limit memory spike
  preds <- predict_chunked(rf_model, test_set, chunk_size = 50000L)

  # Direct keyed update (in-place, no full-column scan)
  pred_db[.(yr), consolidated := preds]

  # Free intermediate objects
  rm(test_set, preds)
  gc()

  cat(sprintf("  Done â€“ %d cells predicted.\n", length(preds)))
}

# â”€â”€ 5. Write output (fwrite is ~5-10Ã— faster than write.csv) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
data.table::fwrite(pred_db, "RF_imputated_db.csv")

cat("All predictions written to RF_imputated_db.csv\n")
```

---

### Summary of Changes

| # | What changed | Why |
|---|---|---|
| 1 | Removed `st_read()` of unused shapefile | Saves potentially gigabytes of RAM |
| 2 | Removed unnecessary library loads | Faster startup, smaller footprint |
| 3 | Fixed `filter(year == year)` â†’ keyed `joined_data[.(yr)]` | Eliminates the variable-masking bug **and** is faster |
| 4 | Switched to `data.table` with `setkey()` | Subsetting and in-place update are orders of magnitude faster |
| 5 | Added `predict_chunked()` | Prevents a single massive allocation when a year has hundreds of thousands of rows |
| 6 | `rm()` + `gc()` after each year | Releases memory between iterations |
| 7 | `fwrite()` instead of `write.csv()` | Much faster serialization for large tables |

These changes should make the script feasible on a personal computer with moderate RAM (8â€“16 GB) without retraining any model.
END_RESPONSE>>>
