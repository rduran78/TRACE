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
run_id: Anthropic_F01_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F01_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original code:

1. **Variable Shadowing Bug:** `filter(year == year)` inside the loop compares the column `year` to itself (due to the loop variable sharing the same name), returning *all* rows instead of the intended subset. This means every iteration processes the entire datasetâ€”massively inflating memory use and computation time.

2. **Unnecessary Spatial Data Load:** `prep_data` is loaded via `st_read` but never used in the prediction loop, wasting memory (potentially gigabytes for a large shapefile).

3. **Entire Dataset in Memory as a `data.frame`:** Both `pred_db` and `joined_data` likely sit in memory as standard data frames. With hundreds of thousands of rows and many predictors, this is inefficient for subsetting and assignment.

4. **Column Assignment in a Loop on a Large Data Frame:** Repeated `pred_db$consolidated[pred_db$year == year] <- ...` triggers full-column scans and copy-on-modify behavior in base R data frames.

5. **Heavy Library Loading:** Several libraries (`blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, etc.) are loaded but unused during prediction, consuming memory.

6. **Writing a Potentially Huge CSV:** `write.csv` on a very large data frame is slow and produces large files without compression.

---

## Optimization Strategy

| Issue | Solution |
|---|---|
| Variable shadowing bug | Rename the loop variable (e.g., `yr`) so `filter` works correctly |
| Unused spatial data load | Remove `st_read` call |
| Unused libraries | Load only what is needed for prediction |
| Inefficient data structure | Convert to `data.table` for fast keyed subsetting and in-place assignment |
| Large CSV output | Use `fwrite` (fast, multi-threaded) with optional compression |
| Optional: memory pressure from predictors | Subset `joined_data` to only the columns the model actually needs before predicting |

---

## Optimized R Code

```r
# â”€â”€ Load only the libraries needed for prediction â”€â”€
library(randomForest)
library(data.table)

# â”€â”€ Load the saved models and associated data â”€â”€
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')

# â”€â”€ Convert to data.table for fast keyed operations â”€â”€
setDT(pred_db)
setDT(joined_data)

# â”€â”€ Identify predictor columns the models actually use â”€â”€
# (Reduces memory passed to predict and avoids copying unnecessary columns)
example_model <- rf_models_per_year[[names(rf_models_per_year)[1]]]
model_vars    <- example_model$forest$xlevels |> names()
# Fallback: if xlevels is empty (all numeric), use rownames of importance
if (length(model_vars) == 0) {
  model_vars <- rownames(example_model$importance)
}
# Keep only needed columns plus the key
keep_cols <- intersect(names(joined_data), c("year", model_vars))
joined_data <- joined_data[, ..keep_cols]

# â”€â”€ Set keys for fast subsetting â”€â”€
setkey(pred_db, year)
setkey(joined_data, year)

# â”€â”€ Pre-allocate the output column â”€â”€
pred_db[, consolidated := NA_real_]

# â”€â”€ Available model years â”€â”€
model_years <- names(rf_models_per_year)

# â”€â”€ Prediction loop (correctly filtered, in-place assignment) â”€â”€
for (yr in unique(pred_db$year)) {
  cat(paste0("Predicting for year ", yr, "\n"))

  yr_char <- as.character(yr)

  if (yr_char %in% model_years) {
    # Fast keyed subset â€” no full-table scan
    test_set <- joined_data[.(yr)]

    rf_model <- rf_models_per_year[[yr_char]]

    preds <- predict(rf_model, newdata = test_set)

    # In-place update by reference â€” no copy-on-modify
    pred_db[.(yr), consolidated := preds]
  } else {
    cat(paste0("Warning: No model found for year ", yr, "\n"))
  }

  # Optional: free memory after each year
  gc()
}

# â”€â”€ Fast, compressed output â”€â”€
fwrite(pred_db, "RF_imputed_db.csv")

cat("Done.\n")
```

---

## Summary of Improvements

| Aspect | Before | After |
|---|---|---|
| **Correctness** | `filter(year == year)` selects ALL rows every iteration | `joined_data[.(yr)]` selects only the target year |
| **Memory** | ~10+ unused libraries + large shapefile loaded | Only 2 libraries; no shapefile; predictor columns trimmed |
| **Subsetting speed** | Full-column logical scan each iteration | O(log n) keyed lookup via `data.table` |
| **Assignment speed** | Copy-on-modify each iteration | In-place `:=` assignment by reference |
| **Output speed** | Base `write.csv` (single-threaded, slow) | `fwrite` (multi-threaded, ~10Ã—+ faster) |
| **Model integrity** | Preserved (not retrained) | Preserved (not retrained) |
END_RESPONSE>>>
