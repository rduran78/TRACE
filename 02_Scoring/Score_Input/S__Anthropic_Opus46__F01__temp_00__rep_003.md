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
run_id: Anthropic_F01_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F01_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, and others are loaded but never used in the prediction phase. Each consumes memory.

2. **Loading a large spatial object unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction loop. This wastes significant memory, especially for a large shapefile.

3. **Variable masking bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which causes `dplyr` to compare the column to itself (always `TRUE`), meaning `test_set` equals the **entire** `joined_data` every iteration. This is both incorrect and extremely expensive.

4. **Whole-dataframe prediction per iteration:** Even if the filter worked correctly, calling `predict()` on hundreds of thousands of rows with a large Random Forest model is memory-intensive. There is no chunking.

5. **Using a `data.frame` for row-level assignment in a loop:** Assigning into `pred_db$consolidated[pred_db$year == year]` repeatedly on a very large data frame is slow due to repeated logical indexing and potential memory copying.

6. **`joined_data` vs `pred_db` ambiguity:** The prediction features come from `joined_data`, but results are stored in `pred_db`. If these are separate large objects both in memory, that doubles RAM usage.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Unused libraries and objects | Remove them to free memory |
| Variable masking bug in `filter()` | Use `.env$year` or rename the loop variable |
| Entire dataset loaded every iteration | Fix the filter so only one year's data is processed |
| Large predict calls | Chunk predictions within each year if still too large |
| Slow data.frame assignment | Convert to `data.table` for fast update-by-reference |
| Two large objects in memory | Unify or minimize duplication; use `gc()` after large operations |
| Writing a huge CSV | Use `data.table::fwrite()` for speed |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries actually needed for prediction â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load the saved models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# Remove any large objects loaded from the .RData that are NOT needed
# (Adjust object names based on what is actually in the .RData file.)
# For example, if prep_data or other training objects were saved:
# rm(prep_data, training_data, ...)
gc()

# â”€â”€ Convert both data frames to data.table for speed â”€â”€
setDT(pred_db)
setDT(joined_data)

# Initialize the output column
pred_db[, consolidated := NA_real_]

# â”€â”€ Get the years that have a trained model â”€â”€
available_years <- names(rf_models_per_year)

# â”€â”€ Set a chunk size to limit peak memory during predict() â”€â”€
CHUNK_SIZE <- 50000L # adjust based on available RAM

# â”€â”€ Prediction loop â”€â”€
unique_years <- unique(pred_db$year)

for (yr in unique_years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (!(yr_char %in% available_years)) {
    cat(paste0("  Warning: No model found for year ", yr, "\n"))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Correctly subset to the current year (avoids the masking bug)
  test_idx <- which(joined_data$year == yr)
  pred_idx <- which(pred_db$year == yr)

  n <- length(test_idx)

  if (n == 0L) {
    cat(paste0("  Warning: No rows in joined_data for year ", yr, "\n"))
    next
  }

  # Predict in chunks to control peak memory
  preds <- numeric(n)
  starts <- seq(1L, n, by = CHUNK_SIZE)

  for (s in starts) {
    e <- min(s + CHUNK_SIZE - 1L, n)
    chunk <- joined_data[test_idx[s:e], ]
    preds[s:e] <- predict(rf_model, newdata = chunk)
  }

  # Fast update by reference
  set(pred_db, i = pred_idx, j = "consolidated", value = preds)

  # Free memory after each year
  rm(chunk, preds)
  gc()
}

# â”€â”€ Write output efficiently â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")

cat("Done.\n")
```

---

## Summary of Key Changes

- **Bug fix:** Replaced `filter(year == year)` with explicit `which(joined_data$year == yr)` using a distinctly named loop variable (`yr`), eliminating the self-comparison that caused every year to process the entire dataset.
- **Memory reduction:** Removed all unused library imports and the unused `st_read()` call; added `gc()` calls.
- **Chunked prediction:** Predictions are made in configurable chunks (`CHUNK_SIZE`) so that `predict()` never needs to allocate output for the full year at once.
- **`data.table` for speed:** Uses `set()` for in-place column updates (no copy-on-modify) and `fwrite()` for fast CSV output.
- **Models are preserved:** No retraining occurs; `rf_models_per_year` is used as-is.
END_RESPONSE>>>
