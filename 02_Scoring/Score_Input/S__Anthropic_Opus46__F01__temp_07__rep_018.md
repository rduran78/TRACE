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
run_id: Anthropic_F01_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary heavy library loading:** Libraries like `blockCV`, `LongituRF`, `spdep`, `plm`, `fixest`, `scales`, `stringi`, and `terra` are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large spatial shapefile (`prep_data`) that is never used** in the prediction section. This can consume gigabytes of RAM for no reason.

3. **Variable shadowing bug:** `filter(year == year)` inside the loop uses the column name `year` on both sides, which means it does **not** filter correctly â€” it evaluates to `TRUE` for every row, so `test_set` equals the entire `joined_data` every iteration. This causes the model to predict on the full dataset every year, massively inflating computation time and producing wrong results.

4. **Predicting on the full dataframe at once** for each year can spike memory, especially with a large Random Forest model and hundreds of thousands of rows with many predictors.

5. **Writing the entire result with `write.csv`** is slower than alternatives like `fwrite` from `data.table`.

6. **No garbage collection or memory management** between yearly iterations.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries and objects | Remove them or don't load them |
| Unused `prep_data` shapefile | Remove the `st_read` call |
| Variable shadowing in `filter()` | Use `.env$year` or rename the loop variable |
| Full-data prediction per iteration | Fix the filter so only the relevant year's rows are predicted |
| Memory pressure during prediction | Predict in chunks if needed; call `gc()` between years |
| Slow CSV write | Use `data.table::fwrite()` |
| Storing predictions | Use `data.table` for fast row-subset assignment |

---

## Optimized R Code

```r
# â”€â”€ Load only what is needed â”€â”€
library(randomForest)
library(data.table)
library(tidyverse)      # kept only if joined_data / pred_db are tibbles

# â”€â”€ Load pre-trained models and prediction database â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# NOTE: Do NOT load the shapefile â€” it is unused in prediction.
# prep_data <- st_read(...)   # REMOVED

# â”€â”€ Convert to data.table for fast subset assignment â”€â”€
setDT(pred_db)
setDT(joined_data)

pred_db[, consolidated := NA_real_]

# â”€â”€ Get the years that actually have a trained model â”€â”€
available_years <- names(rf_models_per_year)
unique_years    <- as.character(unique(pred_db$year))
years_to_predict <- intersect(unique_years, available_years)

cat(paste0("Years with models: ", paste(years_to_predict, collapse = ", "), "\n"))

# â”€â”€ Predict year-by-year â”€â”€
for (yr in years_to_predict) {

  cat(paste0("Predicting for year ", yr, " ... "))

  # Correct filter â€” no variable shadowing
  test_set <- joined_data[year == as.numeric(yr)]

  if (nrow(test_set) == 0L) {
    cat("no data, skipping.\n")
    next
  }

  rf_model <- rf_models_per_year[[yr]]

  # --- Optional: chunk-based prediction to limit peak memory ---
  chunk_size <- 50000L
  n          <- nrow(test_set)
  preds      <- numeric(n)

  if (n <= chunk_size) {
    preds <- predict(rf_model, newdata = test_set)
  } else {
    starts <- seq(1L, n, by = chunk_size)
    for (i in seq_along(starts)) {
      idx <- starts[i]:min(starts[i] + chunk_size - 1L, n)
      preds[idx] <- predict(rf_model, newdata = test_set[idx, ])
    }
  }

  # Fast assignment by reference
  pred_db[year == as.numeric(yr), consolidated := preds]

  cat(paste0(n, " cells done.\n"))

  # Free memory
  rm(test_set, preds)
  gc()
}

# Report years without a model
missing <- setdiff(unique_years, available_years)
if (length(missing) > 0) {
  cat(paste0("Warning: No model for year(s): ", paste(missing, collapse = ", "), "\n"))
}

# â”€â”€ Write output efficiently â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done. Output written to RF_imputated_db.csv\n")
```

---

## Summary of Changes

- **Removed** unused libraries and the unused shapefile load â†’ frees GBs of RAM.
- **Fixed the critical filter bug** (`year == year` â†’ `year == as.numeric(yr)`) so each iteration only processes the correct subset.
- **Switched to `data.table`** for fast by-reference column assignment and subsetting.
- **Added chunked prediction** (configurable `chunk_size`) to cap peak memory during `predict()`.
- **Added `gc()`** after each year to reclaim memory.
- **Replaced `write.csv` with `fwrite`** for significantly faster output.
- **Did not retrain** any model â€” all original `rf_models_per_year` objects are used as-is.
END_RESPONSE>>>
