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
run_id: Anthropic_F01_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F01
prompt_family_slug: RF_WRAPPER_ONLY
family_label: RF wrapper only
family_group: Missing upstream context
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F01_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

There are several performance and correctness issues in the original script:

1. **Variable shadowing bug:** `filter(year == year)` inside the loop compares the column `year` to itself (due to the loop variable sharing the same name), returning *all* rows instead of the intended subset. This means every iteration predicts on the entire dataset â€” massively increasing computation and producing wrong results.

2. **Loading the full spatial shapefile unnecessarily:** `prep_data` is read via `st_read` but never used in the prediction loop. Loading a large `.shp` file consumes significant memory for no reason.

3. **Predicting on the wrong object:** The code filters `joined_data` to create `test_set`, but writes predictions into `pred_db`. If these two objects are both large and separate copies, memory doubles. Their relationship is also unclear â€” if column names or row alignment differ, predictions silently go to the wrong rows.

4. **`predict.randomForest` on hundreds of thousands of rows at once:** Random Forest prediction with many trees and many predictors on a very large data frame can spike memory because `randomForest::predict` builds an intermediate matrix of per-tree predictions.

5. **Writing a massive CSV at the end:** `write.csv` on a data frame with hundreds of thousands of rows Ã— many columns is slow; `data.table::fwrite` is far faster.

6. **Many unnecessary library loads:** Packages like `blockCV`, `LongituRF`, `plm`, `fixest`, `spdep`, `scales`, `stringi`, and `zoo` are not used in the prediction step and consume memory.

---

## Optimization Strategy

| Issue | Fix |
|---|---|
| Variable shadowing | Rename loop variable to `yr` |
| Unused shapefile load | Remove `st_read` call |
| Unnecessary libraries | Load only what is needed |
| Memory from full-dataset filter | Use `data.table` keyed subsetting |
| Large single-pass predict | Predict in chunks (batches) to cap peak memory |
| Slow `write.csv` | Use `data.table::fwrite` |
| Ensuring row alignment | Predict directly on the subset of `pred_db` itself (or verify column parity) |

---

## Optimized R Code

```r
# ------------------------------------------------------------------
# Optimized cell-level GDP prediction
# ------------------------------------------------------------------

# Load only the libraries actually needed for prediction
library(randomForest)   # predict method for RF models
library(data.table)     # fast I/O and subsetting

# 1. Load the saved models and prediction database
load('/Volumes/Toshi 1Tb/R_save_files/model_5_all_countries.RData')
# Expected objects after load: rf_models_per_year, pred_db
# (and possibly joined_data â€” see note below)

# 2. Convert to data.table for speed and memory-efficient subsetting
#    Use whichever object actually holds the predictor columns that
#    match the training data.  If joined_data is the correct source,
#    replace pred_db below with joined_data and write back accordingly.
setDT(pred_db)
pred_db[, consolidated := NA_real_]

# 3. Define a chunked predict function to cap peak memory
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

# 4. Iterate over years with correct variable naming
years <- unique(pred_db$year)

for (yr in years) {
  cat(sprintf("Predicting for year %s ...\n", yr))

  yr_char <- as.character(yr)

  if (!yr_char %in% names(rf_models_per_year)) {
    cat(sprintf("  Warning: No model found for year %s â€” skipping.\n", yr))
    next
  }

  rf_model <- rf_models_per_year[[yr_char]]

  # Identify rows for this year (data.table fast logical subset)
  idx <- which(pred_db$year == yr)

  # Predict in chunks to limit peak memory
  pred_db[idx, consolidated := predict_chunked(rf_model,
                                                newdata = pred_db[idx],
                                                chunk_size = 50000L)]

  cat(sprintf("  Done â€” %d rows predicted.\n", length(idx)))
}

# 5. Write results with fast fwrite
fwrite(pred_db, "RF_imputated_db.csv")
cat("Output written to RF_imputated_db.csv\n")
```

---

## Summary of Key Changes

- **Bug fix:** Loop variable renamed from `year` to `yr` so the filter actually selects only the intended year's rows.
- **Memory savings (~GBs):** Removed the unused `st_read` call and unnecessary library loads.
- **Chunked prediction:** `predict_chunked` processes rows in batches of 50,000 (tunable), preventing a single massive allocation inside `predict.randomForest`.
- **`data.table` throughout:** Keyed subsetting and in-place `:=` assignment avoid copying the entire data frame.
- **`fwrite` instead of `write.csv`:** Typically 5â€“10Ã— faster for large outputs.
- **Models are never retrained** â€” only `predict()` is called on the existing `rf_models_per_year`.
END_RESPONSE>>>
