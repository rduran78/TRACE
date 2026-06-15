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
run_id: Anthropic_F01_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Unnecessary library loading:** Libraries like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and others are loaded but never used in the prediction loop. Each consumes memory.

2. **Loading a large shapefile unnecessarily:** `prep_data` is read via `st_read()` but never used in the prediction workflow. Shapefiles with hundreds of thousands of geometries consume significant RAM.

3. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (due to the loop variable sharing the same name), meaning `test_set` always equals the entire `joined_data` dataframe â€” every year predicts on *all* rows, massively inflating computation time and producing wrong results.

4. **Monolithic `predict()` call on huge data:** Calling `predict()` on hundreds of thousands of rows at once with a large Random Forest model can spike memory usage. Batching is safer.

5. **`pred_db` may be a large dataframe held entirely in memory** while also writing a potentially huge CSV at the end.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Unused libraries | Remove them to reduce memory footprint |
| Unused `prep_data` shapefile | Remove the `st_read()` call entirely |
| Variable shadowing (`year == year`) | Rename the loop variable (e.g., `yr`) so `dplyr::filter` works correctly |
| Large single `predict()` calls | Predict in batches (chunks) to control peak memory |
| Memory pressure | Use `gc()` between years; use `data.table` for subsetting instead of `dplyr` |
| Large CSV output | Write results incrementally or use `fwrite()` from `data.table` for speed |

The trained Random Forest models (`rf_models_per_year`) are **preserved and never retrained**.

---

## Optimized R Code

```r
# â”€â”€ Load only necessary libraries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load pre-trained models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert working dataframes to data.table for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 'joined_data' and 'pred_db' are expected to come from the .RData file.
setDT(joined_data)
setDT(pred_db)

pred_db[, consolidated := NA_real_]

# â”€â”€ Batch-predict helper (controls peak memory) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
predict_in_batches <- function(model, newdata, batch_size = 50000L) {
  n <- nrow(newdata)
  if (n == 0L) return(numeric(0))
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
  for (s in starts) {
    e <- min(s + batch_size - 1L, n)
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
  }
  return(preds)
}

# â”€â”€ Main prediction loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- unique(pred_db$year)

for (yr in years) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% names(rf_models_per_year)) {
    # Correct subsetting â€” no variable-shadowing bug
    idx_test <- which(joined_data$year == yr)
    idx_pred <- which(pred_db$year == yr)

    test_set <- joined_data[idx_test, ]

    rf_model <- rf_models_per_year[[yr_char]]
    pred_db[idx_pred, consolidated := predict_in_batches(rf_model, test_set)]

    # Free temporary objects and reclaim memory
    rm(test_set, rf_model)
    gc()
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }
}

# â”€â”€ Write output efficiently â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fwrite(pred_db, "RF_imputated_db.csv")
cat("Done.\n")
```

---

## Summary of Changes

- **Removed** all unused libraries and the unused `st_read()` call â†’ large memory savings.
- **Fixed the shadowing bug** by renaming the loop variable to `yr`, so `joined_data$year == yr` filters correctly.
- **Switched to `data.table`** for fast subsetting via `which()` and in-place assignment via `:=`.
- **Added batch prediction** (`predict_in_batches`) to cap peak memory during `predict()`.
- **Added `rm()` + `gc()`** after each year to release memory promptly.
- **Replaced `write.csv` with `fwrite`** for significantly faster output.
- **Models are never retrained**; only `predict()` is called on existing `rf_models_per_year`.
END_RESPONSE>>>
